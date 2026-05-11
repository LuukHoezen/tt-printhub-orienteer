import os
import tempfile
import struct
import math
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)

def parse_stl(data):
    """Parse binary STL and return vertices as triangles."""
    if len(data) < 84:
        raise ValueError("Bestand te klein voor STL")
    
    num_triangles = struct.unpack_from('<I', data, 80)[0]
    expected_size = 84 + num_triangles * 50
    
    if len(data) != expected_size:
        raise ValueError("ASCII STL wordt niet ondersteund, gebruik binary STL")
    
    triangles = []
    offset = 84
    for _ in range(num_triangles):
        normal = struct.unpack_from('<fff', data, offset)
        v1 = struct.unpack_from('<fff', data, offset + 12)
        v2 = struct.unpack_from('<fff', data, offset + 24)
        v3 = struct.unpack_from('<fff', data, offset + 36)
        triangles.append((normal, v1, v2, v3))
        offset += 50
    
    return triangles

def rotate_vertices(triangles, axis, angle):
    """Rotate all vertices around an axis by angle (radians)."""
    c, s = math.cos(angle), math.sin(angle)
    ax, ay, az = axis
    
    def rotate_point(p):
        x, y, z = p
        dot = ax*x + ay*y + az*z
        cx = x*c + (ay*z - az*y)*s + ax*dot*(1-c)
        cy = y*c + (az*x - ax*z)*s + ay*dot*(1-c)
        cz = z*c + (ax*y - ay*x)*s + az*dot*(1-c)
        return (cx, cy, cz)
    
    rotated = []
    for normal, v1, v2, v3 in triangles:
        rotated.append((
            rotate_point(normal),
            rotate_point(v1),
            rotate_point(v2),
            rotate_point(v3)
        ))
    return rotated

def translate_to_ground(triangles):
    """Move model so lowest point is at z=0."""
    min_z = min(
        min(v1[2], v2[2], v3[2])
        for _, v1, v2, v3 in triangles
    )
    result = []
    for normal, v1, v2, v3 in triangles:
        result.append((
            normal,
            (v1[0], v1[1], v1[2] - min_z),
            (v2[0], v2[1], v2[2] - min_z),
            (v3[0], v3[1], v3[2] - min_z)
        ))
    return result

def score_orientation(triangles):
    """Score an orientation. Lower is better."""
    total_overhang = 0
    base_contact = 0
    max_z = 0
    
    for normal, v1, v2, v3 in triangles:
        ax, ay, az = v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]
        bx, by, bz = v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]
        cx = ay*bz - az*by
        cy = az*bx - ax*bz
        cz = ax*by - ay*bx
        area = 0.5 * math.sqrt(cx*cx + cy*cy + cz*cz)
        
        length = math.sqrt(cx*cx + cy*cy + cz*cz)
        if length > 1e-10:
            nz = cz / length
        else:
            nz = 0
        
        max_z = max(max_z, v1[2], v2[2], v3[2])
        min_face_z = min(v1[2], v2[2], v3[2])
        
        if nz < -0.5:
            total_overhang += area * abs(nz)
        
        if min_face_z < 0.1 and nz < -0.7:
            base_contact += area
    
    score = total_overhang * 2.0 + max_z * 0.5 - base_contact * 3.0
    return score

def orient_stl(triangles):
    """Find best orientation by testing rotations around key axes."""
    best_score = float('inf')
    best_triangles = triangles
    
    angles = [0, math.pi/2, math.pi, 3*math.pi/2]
    axes = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ]
    
    for axis in axes:
        for angle in angles:
            rotated = rotate_vertices(triangles, axis, angle)
            grounded = translate_to_ground(rotated)
            score = score_orientation(grounded)
            if score < best_score:
                best_score = score
                best_triangles = grounded
    
    return best_triangles

def write_stl(triangles):
    """Write triangles back to binary STL bytes."""
    header = b'\x00' * 80
    num = len(triangles)
    data = header + struct.pack('<I', num)
    
    for normal, v1, v2, v3 in triangles:
        data += struct.pack('<fff', *normal)
        data += struct.pack('<fff', *v1)
        data += struct.pack('<fff', *v2)
        data += struct.pack('<fff', *v3)
        data += struct.pack('<H', 0)
    
    return data

@app.route('/orient', methods=['POST'])
def orient():
    if 'file' not in request.files:
        return jsonify({'error': 'Geen bestand ontvangen'}), 400
    
    file = request.files['file']
    filename = file.filename or 'model.stl'
    
    if not filename.lower().endswith('.stl'):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        file.save(tmp.name)
        return send_file(tmp.name, as_attachment=True, download_name=filename)
    
    try:
        data = file.read()
        triangles = parse_stl(data)
        oriented = orient_stl(triangles)
        result = write_stl(oriented)
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.stl')
        tmp.write(result)
        tmp.flush()
        
        return send_file(tmp.name, as_attachment=True, download_name=filename, mimetype='application/octet-stream')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

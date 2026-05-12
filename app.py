import os
import sys
import tempfile
import struct
from flask import Flask, request, send_file, jsonify
from Tweaker3 import Tweaker

app = Flask(__name__)

def read_stl(filepath):
    """Lees STL bestand en geef mesh terug."""
    with open(filepath, 'rb') as f:
        data = f.read()
    
    if len(data) < 84:
        raise ValueError(f"Bestand te klein: {len(data)} bytes")
    
    num_triangles = struct.unpack_from('<I', data, 80)[0]
    expected_size = 84 + num_triangles * 50
    
    if len(data) != expected_size:
        raise ValueError(f"Geen binary STL: verwacht {expected_size}, kreeg {len(data)}")
    
    mesh = []
    offset = 84
    for _ in range(num_triangles):
        normal = struct.unpack_from('<fff', data, offset)
        v1 = struct.unpack_from('<fff', data, offset + 12)
        v2 = struct.unpack_from('<fff', data, offset + 24)
        v3 = struct.unpack_from('<fff', data, offset + 36)
        mesh.append([list(v1), list(v2), list(v3)])
        offset += 50
    
    return mesh

def apply_matrix(mesh, matrix):
    """Pas rotatiermatrix toe op alle vertices."""
    import numpy as np
    result = []
    for tri in mesh:
        new_tri = []
        for v in tri:
            rotated = np.dot(matrix, v)
            new_tri.append(rotated.tolist())
        result.append(new_tri)
    return result

def write_stl(mesh, filepath):
    """Schrijf mesh terug naar binary STL."""
    import numpy as np
    header = b'\x00' * 80
    num = struct.pack('<I', len(mesh))
    parts = [header, num]
    
    for tri in mesh:
        v1, v2, v3 = np.array(tri[0]), np.array(tri[1]), np.array(tri[2])
        normal = np.cross(v2 - v1, v3 - v1)
        length = np.linalg.norm(normal)
        if length > 1e-10:
            normal = normal / length
        else:
            normal = np.array([0, 0, 1])
        
        parts.append(struct.pack('<fff', *normal))
        parts.append(struct.pack('<fff', *v1))
        parts.append(struct.pack('<fff', *v2))
        parts.append(struct.pack('<fff', *v3))
        parts.append(struct.pack('<H', 0))
    
    with open(filepath, 'wb') as f:
        f.write(b''.join(parts))

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
        # Sla het bestand op
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix='.stl')
        file.save(tmp_in.name)
        tmp_in.close()

        # Lees mesh
        mesh = read_stl(tmp_in.name)

        # Tweaker-3 oriëntatie bepalen
        tweaker = Tweaker.Tweak(mesh, extended_mode=True, verbose=False)
        matrix = tweaker.rotation_matrix

        # Pas matrix toe
        oriented_mesh = apply_matrix(mesh, matrix)

        # Schrijf georiënteerd STL
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.stl')
        tmp_out.close()
        write_stl(oriented_mesh, tmp_out.name)

        return send_file(
            tmp_out.name,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

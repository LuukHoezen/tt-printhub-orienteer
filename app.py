import os
import tempfile
import struct
import numpy as np
from flask import Flask, request, send_file, jsonify
from tweaker3 import Tweaker

app = Flask(__name__)

def read_stl(filepath):
    """Lees binary STL, geef platte lijst van vertices terug."""
    with open(filepath, 'rb') as f:
        data = f.read()
    if len(data) < 84:
        raise ValueError(f"Bestand te klein: {len(data)} bytes")
    num_triangles = struct.unpack_from('<I', data, 80)[0]
    expected_size = 84 + num_triangles * 50
    if len(data) != expected_size:
        raise ValueError(f"Geen binary STL: verwacht {expected_size}, kreeg {len(data)}")
    vertices = []
    offset = 84
    for _ in range(num_triangles):
        for vi in range(3):
            v = list(struct.unpack_from('<fff', data, offset + 12 + vi * 12))
            vertices.append(v)
        offset += 50
    return vertices

def write_stl(mesh, filepath):
    """Schrijf numpy array (n, 3, 3) naar binary STL."""
    parts = [b'\x00' * 80, struct.pack('<I', len(mesh))]
    for tri in mesh:
        v1, v2, v3 = np.array(tri[0]), np.array(tri[1]), np.array(tri[2])
        normal = np.cross(v2 - v1, v3 - v1)
        length = np.linalg.norm(normal)
        normal = normal / length if length > 1e-10 else np.array([0.0, 0.0, 1.0])
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
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix='.stl')
        file.save(tmp_in.name)
        tmp_in.close()

        # Lees als platte lijst: [[x,y,z], [x,y,z], ...]
        content = read_stl(tmp_in.name)
        num_triangles = len(content) // 3

        # Tweaker-3: geef platte lijst mee
        tweaker = Tweaker.Tweak(
            content,
            extended_mode=True,
            verbose=False,
            show_progress=False
        )

        # Haal rotatiematrix op (.Matrix met hoofdletter M)
        # Haal rotatiematrix op uit results array
        # results formaat: [[orientation, bottom_area, overhang, contour, unprintability, [euler_vec, euler_angle, matrix]], ...]
        rotation_matrix = np.array(tweaker.results[0][5][2])

        # Pas rotatie toe: mesh @ rotation_matrix (zoals Tweaker-3 zelf doet)
        mesh = np.array(content, dtype=np.float64).reshape(num_triangles, 3, 3)
        rotated = np.matmul(mesh, rotation_matrix)

        # Zet laagste punt op z=0
        min_z = rotated[:, :, 2].min()
        rotated[:, :, 2] -= min_z

        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.stl')
        tmp_out.close()
        write_stl(rotated, tmp_out.name)

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

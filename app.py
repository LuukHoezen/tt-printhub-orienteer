import os
import tempfile
import struct
import numpy as np
from flask import Flask, request, send_file, jsonify
from tweaker3 import Tweaker

app = Flask(__name__)

def read_stl(filepath):
    """Lees binary STL en geef vertices terug als platte lijst."""
    with open(filepath, 'rb') as f:
        data = f.read()
    if len(data) < 84:
        raise ValueError(f"Bestand te klein: {len(data)} bytes")
    num_triangles = struct.unpack_from('<I', data, 80)[0]
    expected_size = 84 + num_triangles * 50
    if len(data) != expected_size:
        raise ValueError(f"Geen binary STL: verwacht {expected_size}, kreeg {len(data)}")
    # Platte lijst van vertices: [v1, v2, v3, v1, v2, v3, ...]
    vertices = []
    offset = 84
    for _ in range(num_triangles):
        v1 = list(struct.unpack_from('<fff', data, offset + 12))
        v2 = list(struct.unpack_from('<fff', data, offset + 24))
        v3 = list(struct.unpack_from('<fff', data, offset + 36))
        vertices.append(v1)
        vertices.append(v2)
        vertices.append(v3)
        offset += 50
    return vertices

def write_stl(mesh_array, filepath):
    """Schrijf numpy array (n, 3, 3) terug naar binary STL."""
    parts = [b'\x00' * 80, struct.pack('<I', len(mesh_array))]
    for tri in mesh_array:
        v1, v2, v3 = np.array(tri[0]), np.array(tri[1]), np.array(tri[2])
        normal = np.cross(v2 - v1, v3 - v1)
        length = np.linalg.norm(normal)
        normal = normal / length if length > 1e-10 else np.array([0, 0, 1])
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

        # Lees als platte lijst van vertices
        vertices = read_stl(tmp_in.name)

        # Tweaker-3: geef platte lijst mee, hij doet intern reshape(n/3, 3, 3)
        tweaker = Tweaker.Tweak(
            vertices,
            extended_mode=True,
            verbose=False,
            show_progress=False
        )

        matrix = np.array(tweaker.rotation_matrix)

        # Pas rotatie toe op alle vertices
        num_triangles = len(vertices) // 3
        mesh = np.array(vertices, dtype=np.float64).reshape(num_triangles, 3, 3)
        rotated = np.matmul(mesh, matrix.T)

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

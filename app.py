import os
import tempfile
import struct
import numpy as np
from flask import Flask, request, send_file, jsonify
from tweaker3 import Tweaker

app = Flask(__name__)

def read_stl(filepath):
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

def orient_stl(vertices):
    num_triangles = len(vertices) // 3
    tweaker = Tweaker.Tweak(
        vertices,
        extended_mode=True,
        verbose=False,
        show_progress=False
    )
    axis = np.array(tweaker.rotation_axis)
    angle = tweaker.rotation_angle
    axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-10 else np.array([0, 0, 1])
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    rotation_matrix = np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)  ],
    ])
    mesh = np.array(vertices, dtype=np.float64).reshape(num_triangles, 3, 3)
    rotated = np.matmul(mesh, rotation_matrix.T)
    min_z = rotated[:, :, 2].min()
    rotated[:, :, 2] -= min_z
    return rotated

@app.route('/upload', methods=['POST'])
def upload():
    API_KEY  = os.environ.get('PRINTAGO_API_KEY')
    STORE_ID = os.environ.get('PRINTAGO_STORE_ID')

    if not API_KEY or not STORE_ID:
        return jsonify({'error': 'Server niet geconfigureerd'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Geen bestand ontvangen'}), 400

    file = request.files['file']
    filename = request.form.get('filename', file.filename or 'upload.stl')
    name = request.form.get('name', filename.replace('.stl', ''))

    try:
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix='.stl')
        file.save(tmp_in.name)
        tmp_in.close()

        ext = filename.split('.')[-1].lower()

        # Oriënteer STL bestanden
        if ext == 'stl':
            vertices = read_stl(tmp_in.name)
            rotated = orient_stl(vertices)
            triangles = rotated.tolist()
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.stl')
            tmp_out.close()
            write_stl(triangles, tmp_out.name)
            upload_path = tmp_out.name
        else:
            upload_path = tmp_in.name

        import urllib.request
        import json

        headers = {
            'authorization': f'ApiKey {API_KEY}',
            'x-printago-storeid': STORE_ID,
            'content-type': 'application/json',
        }

        # Stap 1: Signed upload URL ophalen
        signed_req = urllib.request.Request(
            'https://api.printago.io/v1/storage/signed-upload-urls',
            data=json.dumps({'filenames': [filename]}).encode(),
            headers=headers,
            method='POST'
        )
        with urllib.request.urlopen(signed_req) as resp:
            signed_data = json.loads(resp.read())

        path = signed_data['signedUrls'][0]['path']
        upload_url = signed_data['signedUrls'][0]['uploadUrl']

        # Stap 2: Bestand uploaden naar signed URL
        with open(upload_path, 'rb') as f:
            file_data = f.read()

        put_req = urllib.request.Request(
            upload_url,
            data=file_data,
            method='PUT'
        )
        with urllib.request.urlopen(put_req) as resp:
            pass

        # Stap 3: Part aanmaken in Printago
        part_body = json.dumps({
            'name': name,
            'type': 'step' if ext in ['step', 'stp'] else '3mf' if ext == '3mf' else 'stl',
            'description': '',
            'fileUris': [path],
            'parameters': [],
            'printTags': {},
            'overriddenProcessProfileId': None,
        }).encode()

        part_req = urllib.request.Request(
            'https://api.printago.io/v1/parts',
            data=part_body,
            headers=headers,
            method='POST'
        )
        with urllib.request.urlopen(part_req) as resp:
            part_data = json.loads(resp.read())

        return jsonify(part_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

import os
import tempfile
import struct
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from tweaker3 import Tweaker

app = Flask(__name__)
CORS(app)

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
        extended_mode=False,
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
    rotated = np.matmul(mesh, rotation_matrix)
    min_z = rotated[:, :, 2].min()
    rotated[:, :, 2] -= min_z
    return rotated

def get_folder_id(klas, api_key, store_id):
    """Zoek map met klasnaam via GET. Fallback naar map 'KLAS'."""
    headers = {
        'authorization': f'ApiKey {api_key}',
        'x-printago-storeid': store_id,
        'content-type': 'application/json',
    }

    def zoek_map(naam):
        res = requests.get(
            'https://api.printago.io/v1/folders/parts',
            headers=headers,
        )
        if not res.ok:
            return None

        data = res.json()

        # API geeft direct een lijst terug
        if isinstance(data, list):
            for item in data:
                if item.get('name', '').strip() == naam:
                    return item['id']
            return None

        # Of een object met items/data
        items = data.get('items', data.get('data', []))
        for item in items:
            if item.get('name', '').strip() == naam:
                return item['id']

        return None

    folder_id = zoek_map(klas)
    if folder_id:
        return folder_id

    return zoek_map('KLAS ONBEKEND')

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
    name = request.form.get('name', filename.rsplit('.', 1)[0])
    klas = request.form.get('klas', '')
    ext = filename.rsplit('.', 1)[-1].lower()

    printago_headers = {
        'authorization': f'ApiKey {API_KEY}',
        'x-printago-storeid': STORE_ID,
    }

    try:
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
        file.save(tmp_in.name)
        tmp_in.close()

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

        folder_id = get_folder_id(klas, API_KEY, STORE_ID) if klas else None

        signed_res = requests.post(
            'https://api.printago.io/v1/storage/signed-upload-urls',
            headers={**printago_headers, 'content-type': 'application/json'},
            json={'filenames': [filename]}
        )
        if not signed_res.ok:
            return jsonify({'error': f'Signed URL fout: {signed_res.text}'}), signed_res.status_code

        signed_data = signed_res.json()
        path = signed_data['signedUrls'][0]['path']
        upload_url = signed_data['signedUrls'][0]['uploadUrl']

        with open(upload_path, 'rb') as f:
            put_res = requests.put(upload_url, data=f)
        if not put_res.ok:
            return jsonify({'error': f'Upload fout: {put_res.status_code}'}), put_res.status_code

        part_body = {
            'name': name,
            'type': 'step' if ext in ['step', 'stp'] else '3mf' if ext == '3mf' else 'stl',
            'description': '',
            'fileUris': [path],
            'parameters': [],
            'printTags': {},
            'overriddenProcessProfileId': None,
            'folderId': folder_id,
        }

        part_res = requests.post(
            'https://api.printago.io/v1/parts',
            headers={**printago_headers, 'content-type': 'application/json'},
            json=part_body
        )
        if not part_res.ok:
            return jsonify({'error': f'Part fout: {part_res.text}'}), part_res.status_code

        return jsonify(part_res.json()), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

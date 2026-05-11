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
    
    # Check if binary STL
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

def get_face_normals(triangles):
    """Get all unique face normals for orientation candidates."""
    normals = set()
    for normal, v1, v2, v3 in triangles:
        # Compute actual normal from vertices
        ax, ay, az = v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]
        bx, by, bz = v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]
        nx = ay*bz - az*by
        ny = az*bx - ax*bz
        nz = ax*by - ay*bx
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length > 1e-10:
            nx, ny, nz = nx/length, ny/length, nz/length
            normals.add((round(nx,4), round(ny,4), round(nz,4)))
    return list(normals)

def rotate_vertices(triangles, axis, angle):
    """Rotate all vertices around an axis by angle (radians)."""
    c, s = math.cos(angle), math.sin(angle)
    ax, ay, az = axis
    
    def rotate_point(p):
        x, y, z = p
        # Rodrigues rotation formula
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
    """
    Score an orientation. Lower is better.
    Criteria: minimize overhang area, minimize height, maximize base contact.
    """
    total_overhang = 0
    base_contact = 0
    max_z = 0
    
    for normal, v1, v2, v3 in triangles:
        # Triangle area
        ax, ay, az = v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]
        bx, by, bz = v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]
        cx = ay*bz - az*by
        cy = az*bx - ax*bz
        cz = ax*by - ay*bx
        area = 0.5 * math.sqrt(cx*cx + cy*cy + cz*cz)
        
        # Compute actual normal
        length = math.sqrt(cx*cx + cy*cy + cz*cz)
        if length > 1e-10:
            nz = cz / length
        else:
            nz = 0
        
        max_z = max(max_z, v1[2], v2[2], v3[2])
        min_face_z = min(v1[2], v2[2], v3[2])
        
        # Overhang: faces pointing downward (nz < -0.5)
        if nz < -0.5:
            total_overhang += area * abs(nz)
        
        # Base contact: faces at z~0 pointing down
        if min_face_z < 0.1 and nz < -0.7:
            base_contact += area
    
    #

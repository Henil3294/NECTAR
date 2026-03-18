import struct
import json

def convert_sparse_to_json(points3d_path, output_json):

    points = []

    print("Reading:", points3d_path)

    with open(points3d_path,"rb") as f:

        num_points = struct.unpack("<Q", f.read(8))[0]

        print("Total points:", num_points)

        for _ in range(num_points):

            point_id = struct.unpack("<Q", f.read(8))[0]

            xyz = struct.unpack("<ddd", f.read(24))

            rgb = struct.unpack("BBB", f.read(3))

            error = struct.unpack("<d", f.read(8))[0]

            track_length = struct.unpack("<Q", f.read(8))[0]

            f.read(track_length * 8)

            points.append({
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "r": rgb[0],
                "g": rgb[1],
                "b": rgb[2]
            })

    print("Writing JSON:", output_json)

    with open(output_json,"w") as f:
        json.dump(points,f)

    print("Conversion complete")
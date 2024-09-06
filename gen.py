import json
import sys

INPUT = "dragon"
X = 100
Y = 100
Z = 100

gltf = json.loads(open(f"{INPUT}.gltf", "r").read())

flt_max = sys.float_info.max
flt_min = sys.float_info.min

aabb_min = [flt_max, flt_max, flt_max]
aabb_max = [flt_min, flt_min, flt_min]

for n in gltf["nodes"]:
    if "translation" in n: translation = n["translation"]
    else: translation = [0, 0, 0]
    if "scale" in n: scale = n["scale"]
    else: scale = [1, 1, 1]
    for p in gltf["meshes"][n["mesh"]]["primitives"]:
        a = gltf["accessors"][p["attributes"]["POSITION"]]
        aabb_min[0] = min(aabb_min[0], translation[0] + a["min"][0] * scale[0])
        aabb_min[1] = min(aabb_min[1], translation[1] + a["min"][1] * scale[1])
        aabb_min[2] = min(aabb_min[2], translation[2] + a["min"][2] * scale[2])
        aabb_max[0] = max(aabb_max[0], translation[0] + a["max"][0] * scale[0])
        aabb_max[1] = max(aabb_max[1], translation[1] + a["max"][1] * scale[1])
        aabb_max[2] = max(aabb_max[2], translation[2] + a["max"][2] * scale[2])

extent = [(aabb_max[0] - aabb_min[0]) * 1.5, (aabb_max[1] - aabb_min[1]) * 1.5, (aabb_max[2] - aabb_min[2]) * 1.5]

nodes = []
for x in range(X):
    for y in range(Y):
        for z in range(Z):
            for n in gltf["nodes"]:
                name = n["name"]
                if "translation" in n: translation = n["translation"]
                else: translation = [0, 0, 0]
                if "scale" in n: scale = n["scale"]
                else: scale = [1, 1, 1]
                if "rotation" in n: rotation = n["rotation"]
                else: rotation = [0, 0, 0, 1]
                nodes.append({
                    "mesh": n["mesh"],
                    "name": f"{name} {x} {y} {z}",
                    "rotation": rotation,
                    "scale": scale,
                    "translation": [
                        translation[0] + (x - X / 2) * extent[0],
                        translation[1] + (y - Y / 2) * extent[1],
                        translation[2] + (z - Z / 2) * extent[2]
                    ]
                })

gltf["scenes"][0]["nodes"] = [i for i in range(len(nodes))]
gltf["nodes"] = nodes
ret = json.dumps(gltf, indent=2)
open(f"{INPUT}_{X}_{Y}_{Z}.gltf", "w").write(ret)

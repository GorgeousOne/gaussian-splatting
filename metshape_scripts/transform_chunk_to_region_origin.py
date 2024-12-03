import Metashape

def get_inv_region_transform(region):
    print(dir(region.rot))
    center = -region.center
    rot = region.rot.t()

    return Metashape.Matrix([
        [rot[0, 0], rot[0, 1], rot[0, 2], center.x],
        [rot[1, 0], rot[1, 1], rot[1, 2], center.y],
        [rot[2, 0], rot[2, 1], rot[2, 2], center.z],
        [0, 0, 0, 1]
    ])


def inverse_transform_chunk(doc):
    if not doc or not doc.chunk:
        print("No active chunk found!")
        return

    chunk = doc.chunk
    region_transform = get_inv_region_transform(chunk.region)
    Metashape.app.addUndo("Transform chunk to region origin")
    chunk.transform.matrix = region_transform

doc = Metashape.app.document
inverse_transform_chunk(doc)

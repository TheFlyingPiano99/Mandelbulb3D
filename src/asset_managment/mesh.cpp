#include "mesh.h"

void Mesh::prepare()
{
    if (indexBuffer.buffer.valid()) {
        indexBuffer.buffer->upload();
    }
    for (auto& it : vertexDataBuffers) {
        if (it.buffer.valid()) {
            it.buffer->upload();
        }
    }
}

bool Mesh::operator==(const Mesh& that) const
{
    return indexBuffer == that.indexBuffer && vertexDataBuffers == that.vertexDataBuffers;
}

bool Mesh::operator!=(const Mesh& that) const
{
    return !(*this == that);
}

Mesh Mesh::empty(std::string name)
{
    Mesh mh;
    mh.mName = std::move(name);
    return mh;
}

bool Mesh::valid() const
{
    if (!vertexDataBuffers[(int)VertexDataElem::Position].buffer.valid()) return false;
    // animation are not mandatory
    return true;
}

bool Mesh::ready() const
{
    if (!(indexBuffer.buffer->mLocation & ResourceLocation::UploadedBit)) return false;
    for (auto& buf : vertexDataBuffers) {
        if (!(buf.buffer->mLocation & ResourceLocation::UploadedBit)) return false;
    }
    return true;
}

std::string Mesh::typeName() {
    return "Mesh";
}

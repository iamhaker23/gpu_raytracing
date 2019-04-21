#include "OBJLoader.h"


std::vector<vector3d> OBJLoader::m_vVertices = std::vector<vector3d>();
std::vector<vector3d> OBJLoader::m_vNormals = std::vector<vector3d>();
std::vector<vector3d> OBJLoader::m_vTexCoords = std::vector<vector3d>();
std::vector<SObjFace> OBJLoader::m_vFaces = std::vector<SObjFace>();
std::vector<ObjMat> OBJLoader::theMats = std::vector<ObjMat>();
std::vector<Vertex> OBJLoader::m_distinctVerts = std::vector<Vertex>();
std::vector<BVH_BAKE> OBJLoader::m_BVH = std::vector<BVH_BAKE>();

float dot2(vec3 l, vec3 r) {
	return (l[0] * r[0]) + (l[1] * r[1]) + (l[2] * r[2]);
}

void cross(vec3 &out, vec3 a, vec3 b) {
	//cx = aybz - azby
	out[0]= a[1] * b[2] - a[2] * b[1];
	//cx = azbx - axbz
	out[1] = a[2] * b[0] - a[0] * b[2];
	//cx = axby - aybx
	out[2] = a[0] * b[1] - a[1] * b[0];
}

int OBJLoader::loadRawVertexList(const char * fileName, Vertex** vertData) {

	m_distinctVerts.clear();

	if (!OBJLoader::myFileLoader(fileName)){
		throw std::exception("OBJLoader: Failed to load model.");
	}

	//loop over faces, store triangle verts
	for (int face = 0; face < m_vFaces.size(); face++) {

		unsigned int vertAIdx = m_vFaces[face].m_uiVertIdx[0];
		unsigned int vertBIdx = m_vFaces[face].m_uiVertIdx[1];
		unsigned int vertCIdx = m_vFaces[face].m_uiVertIdx[2];

		unsigned int normAIdx = m_vFaces[face].m_uiNormalIdx[0];
		unsigned int normBIdx = m_vFaces[face].m_uiNormalIdx[1];
		unsigned int normCIdx = m_vFaces[face].m_uiNormalIdx[2];

		unsigned int uvAIdx = m_vFaces[face].m_uiTexCoordIdx[0];
		unsigned int uvBIdx = m_vFaces[face].m_uiTexCoordIdx[1];
		unsigned int uvCIdx = m_vFaces[face].m_uiTexCoordIdx[2];
		
		//TODO: use vertex indexing to minimise duplication in vertData
		//Get distinct vertex points
		Vertex v1 = Vertex();
		v1.pos[0] = m_vVertices[vertAIdx].pos[0] * VERT_IMPORT_SCALE;
		v1.pos[1] = m_vVertices[vertAIdx].pos[1] * VERT_IMPORT_SCALE;
		v1.pos[2] = m_vVertices[vertAIdx].pos[2] * VERT_IMPORT_SCALE;

		if (normAIdx < m_vNormals.size()) {
			v1.normal[0] = m_vNormals[normAIdx].pos[0];
			v1.normal[1] = m_vNormals[normAIdx].pos[1];
			v1.normal[2] = m_vNormals[normAIdx].pos[2];
		}

		v1.color[0] = 0.5f;
		v1.color[1] = 0.5f;
		v1.color[2] = 0.5f;

		if (uvAIdx < m_vTexCoords.size()) {
			//v1.color[0] = 0.1f * (m_vTexCoords[uvAIdx].pos[0]);
			//v1.color[1] = 0.1f * (m_vTexCoords[uvAIdx].pos[1]);

			v1.uv[0] = (m_vTexCoords[uvAIdx].pos[0]);
			v1.uv[1] = (m_vTexCoords[uvAIdx].pos[1]);
		}

		Vertex v2 = Vertex();
		v2.pos[0] = m_vVertices[vertBIdx].pos[0] * VERT_IMPORT_SCALE;
		v2.pos[1] = m_vVertices[vertBIdx].pos[1] * VERT_IMPORT_SCALE;
		v2.pos[2] = m_vVertices[vertBIdx].pos[2] * VERT_IMPORT_SCALE;

		if (normBIdx < m_vNormals.size()) {
			v2.normal[0] = m_vNormals[normBIdx].pos[0];
			v2.normal[1] = m_vNormals[normBIdx].pos[1];
			v2.normal[2] = m_vNormals[normBIdx].pos[2];
		}
		v2.color[0] = 0.5f;
		v2.color[1] = 0.5f;
		v2.color[2]  = 0.5f;

		if (uvBIdx < m_vTexCoords.size()) {
			//v2.color[0] = 0.1f * (m_vTexCoords[uvBIdx].pos[0]);
			//v2.color[1] = 0.1f * (m_vTexCoords[uvBIdx].pos[1]);


			v2.uv[0] = (m_vTexCoords[uvBIdx].pos[0]);
			v2.uv[1] = (m_vTexCoords[uvBIdx].pos[1]);
		}

		Vertex v3 = Vertex();
		v3.pos[0] = m_vVertices[vertCIdx].pos[0] * VERT_IMPORT_SCALE;
		v3.pos[1] = m_vVertices[vertCIdx].pos[1] * VERT_IMPORT_SCALE;
		v3.pos[2] = m_vVertices[vertCIdx].pos[2] * VERT_IMPORT_SCALE;

		if (normCIdx < m_vNormals.size()) {
			v3.normal[0] = m_vNormals[normCIdx].pos[0];
			v3.normal[1] = m_vNormals[normCIdx].pos[1];
			v3.normal[2] = m_vNormals[normCIdx].pos[2];
		}
		v3.color[0] = 0.5f;
		v3.color[1] = 0.5f;
		v3.color[2] = 0.5f;

		if (uvCIdx < m_vTexCoords.size()) {
			//v3.color[0] = 0.1f * (m_vTexCoords[uvCIdx].pos[0]);
			//v3.color[1] = 0.1f * (m_vTexCoords[uvCIdx].pos[1]);

			v3.uv[0] = (m_vTexCoords[uvCIdx].pos[0]);
			v3.uv[1] = (m_vTexCoords[uvCIdx].pos[1]);
		}

		//Compute tangent and bitangent space
		// Edges of the triangle : position delta
		vec3 deltaPos1 = { 0 };
		deltaPos1[0] = v2.pos[0] - v1.pos[0];
		deltaPos1[1] = v2.pos[1] - v1.pos[1];
		deltaPos1[2] = v2.pos[2] - v1.pos[2];
		vec3 deltaPos2 = { 0 };
		deltaPos2[0] = v3.pos[0] - v1.pos[0];
		deltaPos2[1] = v3.pos[1] - v1.pos[1];
		deltaPos2[2] = v3.pos[2] - v1.pos[2];

		// UV delta
		vec3 deltaUV1 = { 0 };
		deltaUV1[0] = v2.uv[0] - v1.uv[0];
		deltaUV1[0] = v2.uv[1] - v1.uv[1];
		vec3 deltaUV2 = { 0 };
		deltaUV2[0] = v3.uv[0] - v1.uv[0];
		deltaUV2[0] = v3.uv[1] - v1.uv[1];

		float r = 1.0f / (deltaUV1[0] * deltaUV2[1] - deltaUV1[1] * deltaUV2[0]);
		vec3 tangent = { 0 };
		tangent[0] = (deltaPos1[0] * deltaUV2[1] - deltaPos2[0] * deltaUV1[1])*r;
		tangent[1] = (deltaPos1[1] * deltaUV2[1] - deltaPos2[1] * deltaUV1[1])*r;
		tangent[2] = (deltaPos1[2] * deltaUV2[1] - deltaPos2[2] * deltaUV1[1])*r;
		vec3 bitangent = { 0 };
		bitangent[0] = (deltaPos2[0] * deltaUV1[0] - deltaPos1[0] * deltaUV2[0])*r;
		bitangent[1] = (deltaPos2[1] * deltaUV1[0] - deltaPos1[1] * deltaUV2[0])*r;
		bitangent[2] = (deltaPos2[2] * deltaUV1[0] - deltaPos1[2] * deltaUV2[0])*r;

		v1.tangent[0] = tangent[0];
		v1.tangent[1] = tangent[1];
		v1.tangent[2] = tangent[2];
		v1.bitangent[0] = bitangent[0];
		v1.bitangent[1] = bitangent[1];
		v1.bitangent[2] = bitangent[2];

		// Gram-Schmidt orthogonalize
		//ensure axis are orthogonal...
		float dotnt = dot2(v1.normal, tangent);
		tangent[0] = tangent[0] - v1.normal[0] * dotnt;
		tangent[1] = tangent[1] - v1.normal[1] * dotnt;
		tangent[2] = tangent[2] - v1.normal[2] * dotnt;
		float tanmag = sqrtf(tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]);
		tangent[0] /= tanmag; 
		tangent[1] /= tanmag;
		tangent[2] /= tanmag;

		//handedness
		vec3 crossnt = { 0 };
		cross(crossnt, v1.normal, tangent);
		if (dot2(crossnt, bitangent) < 0.0f) {
			tangent[0] = tangent[0] * -1.0f;
			tangent[1] = tangent[1] * -1.0f;
			tangent[2] = tangent[2] * -1.0f;
		}

		/*
		v2.tangent[0] = tangent[0];
		v2.tangent[1] = tangent[1];
		v2.tangent[2] = tangent[2];
		v2.bitangent[0] = bitangent[0];
		v2.bitangent[1] = bitangent[1];
		v2.bitangent[2] = bitangent[2];

		v3.tangent[0] = tangent[0];
		v3.tangent[1] = tangent[1];
		v3.tangent[2] = tangent[2];
		v3.bitangent[0] = bitangent[0];
		v3.bitangent[1] = bitangent[1];
		v3.bitangent[2] = bitangent[2];
		*/

		m_distinctVerts.push_back(v1);
		m_distinctVerts.push_back(v2);
		m_distinctVerts.push_back(v3);

	}

	int size = m_distinctVerts.size();
	std::cout << "Loading " << size << " vertices (scale=" << VERT_IMPORT_SCALE << "x) from " << fileName << std::endl;

	//return size
	return size;
}

void OBJLoader::loadVertices(Vertex* vertData, int numVerts) {


	//TODO: do I understand this?
	//vertData is now an array of pointers?
	for (int v = 0; v < numVerts; v++) {
		vertData[v] = m_distinctVerts[v];
	}
}


int OBJLoader::countBVHNeeded(Vertex* vertData, int numVerts, BVH** bvhData) {
	//getmin,max bounds
	int maxVert[3] = { -1, -1, -1 };
	int minVert[3] = { -1, -1, -1 };

	for (int i = 0; i < numVerts; i++) {

		if (maxVert[0] == -1 || vertData[i].pos[0] > vertData[maxVert[0]].pos[0]) {
			maxVert[0] = i;
		}
		if (maxVert[1] == -1 || vertData[i].pos[1] > vertData[maxVert[1]].pos[1]) {
			maxVert[1] = i;
		}
		if (maxVert[2] == -1 || vertData[i].pos[2] > vertData[maxVert[2]].pos[2]) {
			maxVert[2] = i;
		}

		if (minVert[0] == -1 || vertData[i].pos[0] < vertData[minVert[0]].pos[0]) {
			minVert[0] = i;
		}
		if (minVert[1] == -1 || vertData[i].pos[1] < vertData[minVert[1]].pos[1]) {
			minVert[1] = i;
		}
		if (minVert[2] == -1 || vertData[i].pos[2] < vertData[minVert[2]].pos[2]) {
			minVert[2] = i;
		}


	}

	//divide each dimension by boxsize

	vec3 bvhSpan = { 0 };
	bvhSpan[0] = vertData[maxVert[0]].pos[0] - vertData[minVert[0]].pos[0];
	bvhSpan[1] = vertData[maxVert[1]].pos[1] - vertData[minVert[1]].pos[1];
	bvhSpan[2] = vertData[maxVert[2]].pos[2] - vertData[minVert[2]].pos[2];

	//pad BVH-space
	vec3 padding = { 0 };
	padding[0] = bvhSpan[0];
	padding[1] = bvhSpan[1];
	padding[2] = bvhSpan[2];

	int maxNumBVH[3] = { 1, 1, 1 };

	while (padding[0] >= BVH_BOX_SIZE) {
		padding[0] -= BVH_BOX_SIZE;
		maxNumBVH[0]++;
	}

	while (padding[1] >= BVH_BOX_SIZE) {
		padding[1] -= BVH_BOX_SIZE;
		maxNumBVH[1]++;
	}

	while (padding[2] >= BVH_BOX_SIZE) {
		padding[2] -= BVH_BOX_SIZE;
		maxNumBVH[2]++;
	}

	//init BVH vert lists
	int totalMaxNumBVH = maxNumBVH[0] * maxNumBVH[1] * maxNumBVH[2];

	std::cout << "Populating " << maxNumBVH[0] << "*" << maxNumBVH[1] << "*" << maxNumBVH[2] << "=" << totalMaxNumBVH << " BVH" << std::endl;

	vec3 bvhpaddedmin = { 0 };
	bvhpaddedmin[0] = vertData[minVert[0]].pos[0] - (padding[0] / 2.0f);
	bvhpaddedmin[1] = vertData[minVert[1]].pos[1] - (padding[1] / 2.0f);
	bvhpaddedmin[2] = vertData[minVert[2]].pos[2] - (padding[2] / 2.0f);

	std::cout << "BVH starting at (" << bvhpaddedmin[0] << "," << bvhpaddedmin[1] << "," << bvhpaddedmin[2] << ")" << std::endl;

	int currBVH = 0;
	for (int x = 0; x <= maxNumBVH[0]; x++) {
		for (int y = 0; y <= maxNumBVH[1]; y++) {
			for (int z = 0; z <= maxNumBVH[2]; z++) {

				m_BVH.push_back(BVH_BAKE());

				m_BVH[currBVH].min[0] = bvhpaddedmin[0] + (x*BVH_BOX_SIZE);
				m_BVH[currBVH].min[1] = bvhpaddedmin[1] + (y*BVH_BOX_SIZE);
				m_BVH[currBVH].min[2] = bvhpaddedmin[2] + (z*BVH_BOX_SIZE);

				m_BVH[currBVH].max[0] = bvhpaddedmin[0] + ((x + 1) * BVH_BOX_SIZE);
				m_BVH[currBVH].max[1] = bvhpaddedmin[1] + ((y + 1) * BVH_BOX_SIZE);
				m_BVH[currBVH].max[2] = bvhpaddedmin[2] + ((z + 1) * BVH_BOX_SIZE);

				//std::cout << "BVHMIN->(" << m_BVH[currBVH].min[0] << "," << m_BVH[currBVH].min[1] << "," << m_BVH[currBVH].min[2] << ")" << std::endl;
				//std::cout << "BVHMAX->(" << m_BVH[currBVH].max[0] << "," << m_BVH[currBVH].max[1] << "," << m_BVH[currBVH].max[2] << ")" << std::endl;

				currBVH++;
			}
		}
	}

	int currentRedirect = -1;
	int chunks = 0;

	int addedTris = 0;

	std::cout << "Verts to BVH:" << numVerts << std::endl;

	bool expanded = 0.0f;

	std::vector<int> addedTriIdxList = std::vector<int>();

	//fill bvh vert lists
	for (int i = 0; i < m_BVH.size(); i++) {

		currentRedirect = -1;
		expanded = 0.0f;

		for (int tri = 0; tri < numVerts; tri += 3) {

			//std::cout << "TESTING " << tri << "-> " << m_BVH[i].min[0] << " < " << vertData[tri + 0].pos[0] << " < " << m_BVH[i].max[0] << std::endl;

			//can expand to fit first triangle which does not entirely fit
			//subsequent triangles must strictly fit
			bool containsV1 = (
				//x1
				vertData[tri + 0].pos[0] >= m_BVH[i].min[0]
				&& vertData[tri + 0].pos[0] <= m_BVH[i].max[0]
				//y1
				&& vertData[tri + 0].pos[1] >= m_BVH[i].min[1]
				&& vertData[tri + 0].pos[1] <= m_BVH[i].max[1]
				//z1
				&& vertData[tri + 0].pos[2] >= m_BVH[i].min[2]
				&& vertData[tri + 0].pos[2] <= m_BVH[i].max[2]
				);
			bool containsV2 = (
				//x1
				vertData[tri + 1].pos[0] >= m_BVH[i].min[0]
				&& vertData[tri + 1].pos[0] <= m_BVH[i].max[0]
				//y1
				&& vertData[tri + 1].pos[1] >= m_BVH[i].min[1]
				&& vertData[tri + 1].pos[1] <= m_BVH[i].max[1]
				//z1
				&& vertData[tri + 1].pos[2] >= m_BVH[i].min[2]
				&& vertData[tri + 1].pos[2] <= m_BVH[i].max[2]
				);
			bool containsV3 = (
				//x1
				vertData[tri + 2].pos[0] >= m_BVH[i].min[0]
				&& vertData[tri + 2].pos[0] <= m_BVH[i].max[0]
				//y1
				&& vertData[tri + 2].pos[1] >= m_BVH[i].min[1]
				&& vertData[tri + 2].pos[1] <= m_BVH[i].max[1]
				//z1
				&& vertData[tri + 2].pos[2] >= m_BVH[i].min[2]
				&& vertData[tri + 2].pos[2] <= m_BVH[i].max[2]
				);


			bool containsPartTri = containsV1 || containsV2 || containsV3;
			bool containsWholeTri = containsV1 && containsV2 && containsV3;

			if (containsPartTri) {

				bool addedAlready = false;

				for (int a = 0; a < addedTriIdxList.size(); a++) {
					if (addedTriIdxList[a] == tri) {
						addedAlready = true;
						break;
					}
				}

				if (!addedAlready) {

					addedTris++;
					//std::cout << "TRI IN BVH!" << std::endl;
					if (m_BVH[i].triIdx.size() < BVH_CHUNK_SIZE) {
						//add to bvh current "chunk"
						m_BVH[i].triIdx.push_back(tri);
						addedTriIdxList.push_back(tri);

						if (!containsWholeTri && expanded < BVH_EXPAND_LIMIT) {
							//expand BVH box if needed
							float maxX = std::max(vertData[tri].pos[0], std::max(vertData[tri + 1].pos[0], vertData[tri + 2].pos[0]));
							if (m_BVH[i].max[0] < maxX) {
								expanded += maxX - m_BVH[i].max[0];
								m_BVH[i].max[0] = maxX;
							}
							float maxY = std::max(vertData[tri].pos[1], std::max(vertData[tri + 1].pos[1], vertData[tri + 2].pos[1]));
							if (m_BVH[i].max[1] < maxY) {
								expanded += maxY - m_BVH[i].max[1];
								m_BVH[i].max[1] = maxY;
							}
							float maxZ = std::max(vertData[tri].pos[2], std::max(vertData[tri + 1].pos[2], vertData[tri + 2].pos[2]));
							if (m_BVH[i].max[2] < maxZ) {
								expanded += maxZ - m_BVH[i].max[2];
								m_BVH[i].max[2] = maxZ;
							}
							float minX = std::min(vertData[tri].pos[0], std::min(vertData[tri + 1].pos[0], vertData[tri + 2].pos[0]));
							if (m_BVH[i].min[0] > minX) {
								expanded += m_BVH[i].min[0] - minX;
								m_BVH[i].min[0] = minX;
							}
							float minY = std::min(vertData[tri].pos[1], std::min(vertData[tri + 1].pos[1], vertData[tri + 2].pos[1]));
							if (m_BVH[i].min[1] > minY) {
								expanded += m_BVH[i].min[1] - minY;
								m_BVH[i].min[1] = minY;
							}
							float minZ = std::min(vertData[tri].pos[2], std::min(vertData[tri + 1].pos[2], vertData[tri + 2].pos[2]));
							if (m_BVH[i].min[2] > minZ) {
								expanded += m_BVH[i].min[2] - minZ;
								m_BVH[i].min[2] = minZ;
							}
						}

					}
					else {
						//add to bvh new chunk
						//TODO: doesn't expand when chunked
						if (currentRedirect < 0 || m_BVH[currentRedirect].triIdx.size() == BVH_CHUNK_SIZE) {
							//new BVH chunk required!
							m_BVH.push_back(BVH_BAKE());
							currentRedirect = m_BVH.size() - 1;
							chunks++;

							m_BVH[currentRedirect].max[0] = m_BVH[i].max[0];
							m_BVH[currentRedirect].max[1] = m_BVH[i].max[1];
							m_BVH[currentRedirect].max[2] = m_BVH[i].max[2];

							m_BVH[currentRedirect].min[0] = m_BVH[i].min[0];
							m_BVH[currentRedirect].min[1] = m_BVH[i].min[1];
							m_BVH[currentRedirect].min[2] = m_BVH[i].min[2];
						}
						//use current redirect BVH chunk
						m_BVH[currentRedirect].triIdx.push_back(tri);
						addedTriIdxList.push_back(tri);
					}
				}
			}
		}
	}

	std::cout << "Added tris to BVH: " << addedTris << std::endl;
	std::cout << "BVH chunks: " << chunks << std::endl;

	//return totalnumber of non-zero-containing bvh
	int numFilledBVH = 0;

	//NOTE: m_BVH may contain more than totalMaxnumBVH due to chunking!
	for (int i = 0; i < m_BVH.size(); i++) {
		if (m_BVH[i].hasVerts()) numFilledBVH++;
	}

	std::cout << "Filled " << numFilledBVH << " boxes from " << m_BVH.size() << " total" << std::endl;

	return numFilledBVH;
}

void OBJLoader::createBVH(BVH* bvhData, int numBVH, Vertex* vertData, int numVerts) {

	int added = 0;
	//NOTE: m_BVH may contain more than totalMaxNumBVH due to chunking!
	for (int i = 0; i < m_BVH.size(); i++) {
		if (m_BVH[i].hasVerts()) {
			
			bvhData[added] = BVH();
			bvhData[added].min[0] = m_BVH[i].min[0];
			bvhData[added].min[1] = m_BVH[i].min[1];
			bvhData[added].min[2] = m_BVH[i].min[2];

			bvhData[added].max[0] = m_BVH[i].max[0];
			bvhData[added].max[1] = m_BVH[i].max[1];
			bvhData[added].max[2] = m_BVH[i].max[2];

			std::cout << "BVH->max(" << bvhData[added].max[0] << "," << bvhData[added].max[1] << "," << bvhData[added].max[2] << ")" << std::endl;
			std::cout << "BVH->min(" << bvhData[added].min[0] << "," << bvhData[added].min[1] << "," << bvhData[added].min[2] << ")" << std::endl;

			int numTrisToAdd = m_BVH[i].triIdx.size();

			std::cout << "Tris to add:" << numTrisToAdd << std::endl;

			if (numTrisToAdd > BVH_CHUNK_SIZE) throw new std::exception("Tried to add too many tris to BVH box");

			bvhData[added].numTris = numTrisToAdd;

			for (int t = 0; t < numTrisToAdd; t++) {
				std::cout << "Adding " << t << "=" << m_BVH[i].triIdx[t] << std::endl;
				bvhData[added].triIdx[t] = m_BVH[i].triIdx[t];
			}

			added++;
		}
	}

	if (added > numBVH) throw new std::exception("Tried to add more BVH than expected");

}


bool OBJLoader::myFileLoader(const char *filename)
{

	m_vVertices.clear();
	m_vFaces.clear();
	m_vNormals.clear();
	m_vTexCoords.clear();

	char line[255];

	int currentMat = -1;

	FILE * theFile;
	fopen_s(&theFile, filename, "rt");
	if (!theFile)
	{
		std::cout << "Can't open the file" << std::endl;
		return false;
	}

	while (!feof(theFile))
	{
		char firstChar = fgetc(theFile);

		switch (firstChar)
		{
		case 'v':   //It's a vertex or vertex attribut
		{
			char secondChar = fgetc(theFile);

			switch (secondChar)
			{
			case ' ':   //space or tab - must be just a vert
			case '\t':
			{
				float thePoints[3];
				fgets(line, 255, theFile); //read in the whole line				
				sscanf_s(line, " %f %f %f", &thePoints[0], &thePoints[1], &thePoints[2]); //get the vertex coords
				
				//Flip when exporting from OBJ to openGL
				//thePoints[0] *= -1.0f;
				thePoints[1] *= -1.0f;
				//thePoints[2] *= -1.0f;

				vector3d tmp = vector3d();
				tmp.pos[0] = thePoints[0];
				tmp.pos[1] = thePoints[1];
				tmp.pos[2] = thePoints[2];

				m_vVertices.push_back(tmp); //add to the vertex array
				break;
			}
			case 'n':
			{
				float theNormals[3];
				fgets(line, 255, theFile); //get the Normals						
				sscanf_s(line, " %f %f %f", &theNormals[0], &theNormals[1], &theNormals[2]); //get the normal coords	

				
				vector3d tmp = vector3d();
				tmp.pos[0] = theNormals[0];
				tmp.pos[1] = theNormals[1];
				tmp.pos[2] = theNormals[2];

				m_vNormals.push_back(tmp); //add to the normal array
				break;
			}
			case 't':
			{
				vec3 theTex = { 0 };
				fgets(line, 255, theFile); //get the Tex						
				sscanf_s(line, " %f %f", &theTex[0], &theTex[1]); //get the texture coords							
				//theTex[1] = 1-theTex[1];		

				vector3d tmp = vector3d();
				tmp.pos[0] = theTex[0];
				tmp.pos[1] = theTex[1];
				tmp.pos[2] = theTex[2];

				m_vTexCoords.push_back(tmp); //add to the text-coord array
				break;
			}
			}
			break;
		}
		case 'f': //Read in a face
		{
			readFaceLine(theFile, currentMat);
			break;
		}
		case 'm': //It's the material lib
		{
			char buff[255];
			char buff2[255];
			fgets(line, 255, theFile); //read in the whole line	

		//	sscanf(line, "%s ./%s", &buff, &buff2);
			sscanf_s(line, "%s %s", &buff, sizeof(char) * 255, &buff2, sizeof(char) * 255);
			myMTLLoader(filename, buff2);
			break;
		}
		case 'u': //Change current Material
		{
			char buff[255];
			char buff2[255];
			fgets(line, 255, theFile); //read in the whole line	

			sscanf_s(line, "%s %s", &buff, sizeof(char) * 255, &buff2, sizeof(char) * 255);

			currentMat = lookupMaterial(buff2);

			break;
		}
		default: // A bit we don't know about - skip line
		{
			if ((firstChar != 10))
			{
				fgets(line, 255, theFile); //read and throw away
			}
		}
		}

	}


	return true;

}

void OBJLoader::splitFrontString(char * inputString, char * frontString, char * restString, int size)
{
	sscanf_s(inputString, "%s", frontString, size);

	//Get length of frontString

	int x = 0;
	while (frontString[x] != '\0') x++;
	if (inputString[0] == ' ') x++;


	int y = 0;
	while (inputString[y + x + 1] != '\0')
	{
		restString[y] = inputString[y + x + 1];
		y++;
	}
	restString[y] = '\0';


}

void OBJLoader::readTriangleFaceVertTexNorm(char *line, int matId)
{
	unsigned int m_uiVertIdx[3];
	unsigned int m_uiTexCoordIdx[3];
	unsigned int m_uiNormalIdx[3];

	for (int x = 0; x < 3; x++)
	{
		char currentArg[255];
		char remStr[255];

		splitFrontString(line, currentArg, remStr, sizeof(char) * 255);

		sscanf_s(currentArg, "%i/%i/%i", &m_uiVertIdx[x], &m_uiTexCoordIdx[x], &m_uiNormalIdx[x]);
		memcpy_s(line, sizeof(char) * 255, remStr, 255);
	}

	SObjFace newFace(m_uiVertIdx, m_uiTexCoordIdx, m_uiNormalIdx, matId);

	m_vFaces.push_back(newFace);
}

void OBJLoader::readFaceLine(FILE * theFile, int matId)
{
	char line[255];
	fgets(line, 255, theFile); //read in the whole line				

	readTriangleFaceVertTexNorm(line, matId);
}

bool OBJLoader::myMTLLoader(const char *mainName, const char *filename)
{
	char line[255];

	std::string fullPath(mainName);

	int whereDash = (int)(fullPath.find_last_of('/'));

	std::string actualPath = fullPath.substr(0, whereDash);

	std::string s = actualPath + std::string("\\") + std::string(filename);

	std::ifstream fin;

	fin.open(s.c_str());

	if (fin.fail())

	{

		std::cout << "Can't open the file" << std::endl;

		return false;

	}

	std::string identifierStr;

	bool foundNewMaterial = false;

	ObjMat newMaterial;

	while (!fin.eof())

	{

		fin >> identifierStr;

		if (identifierStr == "#")

		{

			fin.getline(line, 255);

			fin >> std::ws;

		}
		else if (identifierStr == "newmtl")
		{
			//cout << "newmtl" <<endl;
			if (foundNewMaterial)
			{

				//char buf[255];

				//cout << " newmtl " << newMaterial.matName << " " << newMaterial.textureName << endl;

				//sprintf(buf, "%s/%s",actualPath.c_str(),newMaterial.textureName); 
				std::string s = actualPath + "\\" + newMaterial.textureName;
				//newMaterial.glIndex = TextureHandler::lookUpTexture(s, true);

				if (newMaterial.bump) {
					std::string s2 = actualPath + "\\" + newMaterial.bumpTextureName;
					//newMaterial.glIndexBump = TextureHandler::lookUpTexture(s2, true);
				}

				theMats.push_back(newMaterial);

			}

			fin.getline(line, 255);

			fin >> std::ws;

			sscanf_s(line, "%s", newMaterial.matName, _countof(line));
			std::cout << newMaterial.matName << " MATERIAL NAME>" << std::endl;

			foundNewMaterial = true;

		}

		else if (identifierStr == "Ns" || identifierStr == "Ni" || identifierStr == "Tr" || identifierStr == "Tf")//skip all these

		{

			fin.getline(line, 255);

			fin >> std::ws;

		}

		else if (identifierStr == "d")

		{

			fin.getline(line, 255);

			fin >> std::ws;

			sscanf_s(line, " %f", &newMaterial.d);

		}

		else if (identifierStr == "illum")

		{

			fin.getline(line, 255);

			fin >> std::ws;

			sscanf_s(line, "llum %i", &newMaterial.illum);

		}

		else if (identifierStr == "Ka")

		{

			fin.getline(line, 255); //Ambient

			sscanf_s(line, "%f %f %f", &newMaterial.ambientCol[0], &newMaterial.ambientCol[1], &newMaterial.ambientCol[2]);

		}

		else if (identifierStr == "Kd")

		{

			fin.getline(line, 255); //diffuse

			fin >> std::ws;

			sscanf_s(line, "Kd %f %f %f", &newMaterial.diffuseCol[0], &newMaterial.diffuseCol[1], &newMaterial.diffuseCol[2]);

		}

		else if (identifierStr == "Ks")

		{

			fin.getline(line, 255); //specular

			fin >> std::ws;

			sscanf_s(line, "Ks %f %f %f", &newMaterial.specCol[0], &newMaterial.specCol[1], &newMaterial.specCol[2]);

		}

		else if (identifierStr == "Ke")///not used so skip

		{

			fin.getline(line, 255); //emissive

			fin >> std::ws;

		}

		else if (identifierStr == "map_Kd")

		{

			fin.getline(line, 255); //textureName

			fin >> std::ws;

			sscanf_s(line, "%s", &newMaterial.textureName, _countof(line));

		}

		else if (identifierStr == "map_Bump")//modified for normal map

		{

			fin.getline(line, 255); //textureName

			fin >> std::ws;

			newMaterial.bump = true;

			sscanf_s(line, "%s", &newMaterial.bumpTextureName, _countof(line));

		}

		else//skip anything else

		{

			fin.getline(line, 255);

			fin >> std::ws;

		}

	}

	if (foundNewMaterial)

	{

		//char buf[255];

		//sprintf(buf, "%s/%s",actualPath.c_str(),newMaterial.textureName); 
		std::string s = actualPath + "\\" + newMaterial.textureName;

		//newMaterial.glIndex = TextureHandler::lookUpTexture(s, true);

		if (newMaterial.bump) {
			std::string s2 = actualPath + "\\" + newMaterial.bumpTextureName;
			//newMaterial.glIndexBump = TextureHandler::lookUpTexture(s2, true);
		}

		std::cout << "MATERIAL " << newMaterial.textureName << " " << newMaterial.glIndex << "+" << newMaterial.glIndexBump << std::endl;

		theMats.push_back(newMaterial);

	}

	std::cout << "Number of Materials Loaded " << (int)theMats.size() << std::endl;

	return true;

}

int OBJLoader::lookupMaterial(char *matName)
{
	for (int count = 0; count < (int)theMats.size(); count++)
	{
		if (strcmp(theMats[count].matName, matName) == 0)
		{
			return count;
		}
	}
	return -1;
}
#include "OBJLoader.h"

#define NO_LOSS_BVH 1

std::vector<vector3d> OBJLoader::m_vVertices = std::vector<vector3d>();
std::vector<vector3d> OBJLoader::m_vNormals = std::vector<vector3d>();
std::vector<vector3d> OBJLoader::m_vTexCoords = std::vector<vector3d>();
std::vector<SObjFace> OBJLoader::m_vFaces = std::vector<SObjFace>();
std::vector<ObjMat> OBJLoader::theMats = std::vector<ObjMat>();
std::vector<Vertex> OBJLoader::m_distinctVerts = std::vector<Vertex>();
std::vector<BVH_BAKE> OBJLoader::m_BVH = std::vector<BVH_BAKE>();
std::vector<BVH> OBJLoader::bvhBACKUP = std::vector<BVH>();

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

float max(float a, float b) {
	return (a > b) ? a : b;
}
float min(float a, float b) {
	return (a < b) ? a : b;
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

		//fix: uv co-ordinates are not in sequence order
		unsigned int uvAIdx = m_vFaces[face].m_uiTexCoordIdx[2];
		unsigned int uvBIdx = m_vFaces[face].m_uiTexCoordIdx[0];
		unsigned int uvCIdx = m_vFaces[face].m_uiTexCoordIdx[1];
		
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

		/*
		//precomupted tangent/bitangent ot used by ray-tracer - consider deprecating code
		
		float r = 1.0f / (deltaUV1[0] * deltaUV2[1] - deltaUV1[1] * deltaUV2[0]);
		vec3 tangent = { 0 };
		tangent[0] = (deltaPos1[0] * deltaUV2[1] - deltaPos2[0] * deltaUV1[1])*r;
		tangent[1] = (deltaPos1[1] * deltaUV2[1] - deltaPos2[1] * deltaUV1[1])*r;
		tangent[2] = (deltaPos1[2] * deltaUV2[1] - deltaPos2[2] * deltaUV1[1])*r;
		vec3 bitangent = { 0 };
		bitangent[0] = (deltaPos2[0] * deltaUV1[0] - deltaPos1[0] * deltaUV2[0])*r;
		bitangent[1] = (deltaPos2[1] * deltaUV1[0] - deltaPos1[1] * deltaUV2[0])*r;
		bitangent[2] = (deltaPos2[2] * deltaUV1[0] - deltaPos1[2] * deltaUV2[0])*r;

		//v1.tangent[0] = tangent[0];
		//v1.tangent[1] = tangent[1];
		//v1.tangent[2] = tangent[2];
		//v1.bitangent[0] = bitangent[0];
		//v1.bitangent[1] = bitangent[1];
		//v1.bitangent[2] = bitangent[2];

		// Gram-Schmidt orthogonalize
		// ensure axis are orthogonal...
		float dotnt = dot2(v1.normal, tangent);
		tangent[0] = tangent[0] - v1.normal[0] * dotnt;
		tangent[1] = tangent[1] - v1.normal[1] * dotnt;
		tangent[2] = tangent[2] - v1.normal[2] * dotnt;
		float tanmag = sqrtf(tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]);
		tangent[0] /= tanmag; 
		tangent[1] /= tanmag;
		tangent[2] /= tanmag;

		// handedness
		vec3 crossnt = { 0 };
		cross(crossnt, v1.normal, tangent);
		if (dot2(crossnt, bitangent) < 0.0f) {
			tangent[0] = tangent[0] * -1.0f;
			tangent[1] = tangent[1] * -1.0f;
			tangent[2] = tangent[2] * -1.0f;
		}
		*/

		m_distinctVerts.push_back(v1);
		m_distinctVerts.push_back(v2);
		m_distinctVerts.push_back(v3);

	}

	int size = static_cast<int>(m_distinctVerts.size());
	std::cout << "Loading " << size << " vertices (scale=" << VERT_IMPORT_SCALE << "x) from " << fileName << std::endl;

	return size;
}

void OBJLoader::loadVertices(Vertex* vertData, int numVerts) {

	//vertData is now an array of pointers to vertex objects
	//Created by copy constructor?
	for (int v = 0; v < numVerts; v++) {
		vertData[v] = m_distinctVerts[v];
	}
}

///////////////////////////////////////////////////////
//Linear BVH code from: https://devblogs.nvidia.com/thinking-parallel-part-iii-tree-construction-gpu/
///////////////////////////////////////////////////////

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(float x, float y, float z)
{
	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = expandBits((unsigned int)x);
	unsigned int yy = expandBits((unsigned int)y);
	unsigned int zz = expandBits((unsigned int)z);
	return xx * 4 + yy * 2 + zz;
}

int findSplit(unsigned int* sortedMortonCodes,
	int           first,
	int           last)
{
	// Identical Morton codes => split the range in the middle.

	unsigned int firstCode = sortedMortonCodes[first];
	unsigned int lastCode = sortedMortonCodes[last];

	if (firstCode == lastCode)
		return (first + last) >> 1;

	// Calculate the number of highest bits that are the same
	// for all objects, using the count-leading-zeros intrinsic.
	//NOTE: Adapted for VC++
	unsigned long commonPrefix = -1;
	_BitScanForward(&commonPrefix, firstCode ^ lastCode);

	// Use binary search to find where the next bit differs.
	// Specifically, we are looking for the highest object that
	// shares more than commonPrefix bits with the first one.

	int split = first; // initial guess
	int step = last - first;

	do
	{
		step = (step + 1) >> 1; // exponential decrease
		int newSplit = split + step; // proposed new position

		if (newSplit < last)
		{
			unsigned int splitCode = sortedMortonCodes[newSplit];
			
			//NOTE: Adapted for VC++
			unsigned long splitPrefix = 0l;
			_BitScanForward(&splitPrefix, firstCode ^ splitCode);
			
			if (splitPrefix > commonPrefix)
				split = newSplit; // accept proposal
		}
	} while (step > 1);

	return split;
}

bool OBJLoader::mortonCodeSort(TriangleBounds a, TriangleBounds b) {
	return a.mortonCode > b.mortonCode;
}

void OBJLoader::createLinearBVH(std::vector<vector3d> barycentres, std::vector<float> radii, float* sceneCentroid, float* cubeSize) {

	int num = static_cast<int>(barycentres.size());

	std::vector<TriangleBounds> bounds = std::vector<TriangleBounds>();

	std::vector<int> merged = std::vector<int>();

	for (int i = 0; i < num; i++) {
		
		bool alreadyMerged = false;
		for (int j = 0; j < static_cast<int>(merged.size()); j++) {
			if (merged[j] == i) {
				alreadyMerged = true;
				break;
			}
		}
		if (alreadyMerged) continue;
		
		TriangleBounds a = TriangleBounds();

		//mortoncode relative to scene centroid (and unit size)
		a.centre[0] = (barycentres[i].pos[0]);
		a.centre[1] = (barycentres[i].pos[1]);
		a.centre[2] = (barycentres[i].pos[2]);

		/*

	std::cout << (barycentres[i].pos[0] - sceneCentroid[0]) / cubeSize[0]
		<< ":" << (barycentres[i].pos[1] - sceneCentroid[1]) / cubeSize[1]
		<< ":" << (barycentres[i].pos[2] - sceneCentroid[2] )/ cubeSize[2] << std::endl;
		*/

		a.mortonCode = morton3D((barycentres[i].pos[0] - sceneCentroid[0]) / cubeSize[0], (barycentres[i].pos[1] - sceneCentroid[1]) / cubeSize[1], (barycentres[i].pos[2] - sceneCentroid[2]) / cubeSize[2]);
		
		a.objIds.push_back(i);

		a.radius = radii[i];

		//TODO: allow merging for less deep tree! First, bugfix traversal issues...
		
		/*
		
		for (int x = i+1; x < num; x++) {
			
			bool alreadyMerged2 = false;
			for (int j2 = 0; j2 < static_cast<int>(merged.size()); j2++) {
				if (merged[j2] == x) {
					alreadyMerged2 = true;
					break;
				}
			}
			if (!alreadyMerged2) {

				vec3 newCentre = { 0 };

				newCentre[0] = (a.centre[0] + barycentres[x].pos[0]) / 2;
				newCentre[1] = (a.centre[1] + barycentres[x].pos[1]) / 2;
				newCentre[2] = (a.centre[2] + barycentres[x].pos[2]) / 2;

				float acentredist = ((a.centre[0] - newCentre[0])*(a.centre[0] - newCentre[0]))
					+ ((a.centre[1] - newCentre[1])*(a.centre[1] - newCentre[1]))
					+ ((a.centre[2] - newCentre[2])*(a.centre[2] - newCentre[2]));
				float bcentredist = ((barycentres[x].pos[0] - newCentre[0])*(barycentres[x].pos[0] - newCentre[0]))
					+ ((barycentres[x].pos[1] - newCentre[1])*(barycentres[x].pos[1] - newCentre[1]))
					+ ((barycentres[x].pos[2] - newCentre[2])*(barycentres[x].pos[2] - newCentre[2]));

				float ar = sqrtf(acentredist) + a.radius;
				float br = sqrtf(bcentredist) + radii[x];
				float newRadius = (ar > br) ? ar : br;


				//threshold for merging tris into same bvh
				if ( (newRadius - a.radius) / newRadius <= 0.3f) {
					merged.push_back(x);
					a.objIds.push_back(x);
					a.radius = (a.radius < newRadius) ? newRadius : a.radius;
					a.centre[0] = newCentre[0];
					a.centre[1] = newCentre[1];
					a.centre[2] = newCentre[2];
				}
			}
		}
		*/
		
		

		bounds.push_back(a);
	}

	unsigned int* sortedMortonCodes = new unsigned int[num];
	//int* sortedObjectIds = new int[num];
	std::vector<int>* sortedObjectIds = new std::vector<int>[num];
	float* sortedRadii = new float[num];
	float* sortedCenters = new float[num * 3];

	std::sort(bounds.begin(), bounds.end(), mortonCodeSort);

	for (int i = 0; i < static_cast<int>(bounds.size()); i++) {
		sortedMortonCodes[i] = bounds[i].mortonCode;
		//sortedObjectIds[i] = bounds[i].objId;
		sortedObjectIds[i] = bounds[i].objIds;

		sortedRadii[i] = bounds[i].radius;
		sortedCenters[(i * 3)] = bounds[i].centre[0];
		sortedCenters[(i * 3)+1] = bounds[i].centre[1];
		sortedCenters[(i * 3)+2] = bounds[i].centre[2];
	}

	generateHierarchy(sortedMortonCodes, sortedObjectIds, 0, num-1, sortedRadii, sortedCenters, num, 0);

	std::reverse(m_BVH.begin(), m_BVH.end());

}

BVH_BAKE*  OBJLoader::generateHierarchy(unsigned int* sortedMortonCodes,
	//int*          sortedObjectIDs,
	std::vector<int>* sortedObjectIDs,
	int           first,
	int           last,
	float* radii,
	float* centres,
	int numObj
,int depth)
{
	// Single object => create a leaf node.
	if (first == last) {
		//int objId = (first < numObj && first >= 0) ? sortedObjectIDs[first] : -1;
		//if (objId != -1) {
		if (first < numObj && sortedObjectIDs[first].size() > 0){
			int objId = first;// sortedObjectIDs[first].at(0);
			BVH_BAKE* leaf = new BVH_BAKE(depth, &sortedObjectIDs[first], radii[objId], centres[(objId * 3)], centres[(objId * 3) + 1], centres[(objId * 3) + 2]);
			
			int idx = static_cast<int>(m_BVH.size());
			leaf->idx = idx;
			m_BVH.push_back(*leaf);

			return leaf;
		}
		else {
			BVH_BAKE* leaf = new BVH_BAKE(depth, false);

			int idx = static_cast<int>(m_BVH.size());
			leaf->idx = idx;
			m_BVH.push_back(*leaf);

			return leaf;
		}
	}

	// Determine where to split the range.
	int split = findSplit(sortedMortonCodes, first, last);

	// Process the resulting sub-ranges recursively.
	BVH_BAKE* childA = generateHierarchy(sortedMortonCodes, sortedObjectIDs,
		first, split, radii, centres, numObj, depth+1);

	BVH_BAKE* childB = generateHierarchy(sortedMortonCodes, sortedObjectIDs,
		split + 1, last, radii, centres, numObj, depth+1);
	

	int idx = static_cast<int>(m_BVH.size());

	BVH_BAKE* parent = new BVH_BAKE(depth, childA, childB);
	parent->idx = idx;
	m_BVH.push_back(*parent);

	return parent;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////



int OBJLoader::countBVHNeeded(Vertex* vertData, int numVerts) {


	float padding = 500.0f;
	std::vector<float> radii = std::vector<float>();
	std::vector<vector3d> barycenters = std::vector <vector3d> ();
	
	float* cubeSize = new float[3];
	cubeSize[0] = 0;
	cubeSize[1] = 0;
	cubeSize[2] = 0;

	for (int i = 0; i < numVerts; i += 3) {
		vector3d barycenter = vector3d();

		//NOTE: encoding triangle-vertices indices as barycentre indices
		barycenter.pos[0] = (vertData[i].pos[0] + vertData[i + 1].pos[0] + vertData[i + 2].pos[0]) / 3.0f;
		barycenter.pos[1] = (vertData[i].pos[1] + vertData[i + 1].pos[1] + vertData[i + 2].pos[1]) / 3.0f;
		barycenter.pos[2] = (vertData[i].pos[2] + vertData[i + 1].pos[2] + vertData[i + 2].pos[2]) / 3.0f;
		barycenters.push_back(barycenter);
		
		float distances[3] = { 0 };
		
		distances[0] = ((barycenter.pos[0] - vertData[i].pos[0]) *(barycenter.pos[0] - vertData[i].pos[0]))
					+ ((barycenter.pos[1] - vertData[i].pos[1]) *(barycenter.pos[1] - vertData[i].pos[1]))
					+ ((barycenter.pos[2] - vertData[i].pos[2]) *(barycenter.pos[2] - vertData[i].pos[2]));
		
		distances[1] = ((barycenter.pos[0] - vertData[i+1].pos[0]) *(barycenter.pos[0] - vertData[i + 1].pos[0]))
					+ ((barycenter.pos[1] - vertData[i + 1].pos[1]) *(barycenter.pos[1] - vertData[i + 1].pos[1]))
					+ ((barycenter.pos[2] - vertData[i + 1].pos[2]) *(barycenter.pos[2] - vertData[i + 1].pos[2]));
		
		distances[2] = ((barycenter.pos[0] - vertData[i + 2].pos[0]) *(barycenter.pos[0] - vertData[i + 2].pos[0]))
					+ ((barycenter.pos[1] - vertData[i + 2].pos[1]) *(barycenter.pos[1] - vertData[i + 2].pos[1]))
					+ ((barycenter.pos[2] - vertData[i + 2].pos[2]) *(barycenter.pos[2] - vertData[i + 2].pos[2]));
		
		float radius = sqrtf(max(max(distances[0], distances[1]), distances[2]));// +padding;
		std::cout << "Radius:" << radius << std::endl;
		radii.push_back(radius);

		if (cubeSize[0] < abs(radius) + abs(barycenter.pos[0]*0)) cubeSize[0] = abs(radius) + abs(barycenter.pos[0] * 0);
		if (cubeSize[1] < abs(radius) + abs(barycenter.pos[1] * 0)) cubeSize[1] = abs(radius) + abs(barycenter.pos[1] * 0);
		if (cubeSize[2] < abs(radius) + abs(barycenter.pos[2] * 0)) cubeSize[2] = abs(radius) + abs(barycenter.pos[2] * 0);
	}

	float sceneCentroid[3] = { 0 };
	
	
	for (int i = 0; i < static_cast<int>(barycenters.size()); i++) {
		sceneCentroid[0] += barycenters[i].pos[0];
		sceneCentroid[1] += barycenters[i].pos[1];
		sceneCentroid[2] += barycenters[i].pos[2];
	}
	sceneCentroid[0] /= static_cast<int>(barycenters.size());
	sceneCentroid[1] /= static_cast<int>(barycenters.size());
	sceneCentroid[2] /= static_cast<int>(barycenters.size());
	

	std::cout << "Scene Centroid:" << sceneCentroid[0] << "," << sceneCentroid[1] << "," << sceneCentroid[2] << std::endl;

	createLinearBVH(barycenters, radii, sceneCentroid, cubeSize);

	return static_cast<int>(m_BVH.size());

}

int OBJLoader::getMaxDepth() {
	int maxDepth = 0;

	for (int i = 0; i < static_cast<int>(m_BVH.size()); i++) {
		if (m_BVH[i].depth > maxDepth) {
			maxDepth = m_BVH[i].depth;
		}
	}
	
	std::cout << "Max Depth:" << maxDepth << std::endl;

	return maxDepth;
}

int OBJLoader::putBVH(BVH* bvhData, BVH_BAKE* bvh, Vertex* vertData, int numVerts, int bvhIdx) {
	
	std::cout << "PUT:" << bvh->idx << std::endl;

	//put the BVH_BAKE data into the bvh list at the given index

	int added = 0;

	int front = -1;
	int back = -1;

	//if the list is reversed, the indices are relative to the end
	if (static_cast<int>(bvh->children.size()) == 2 && bvh->children[0] + bvh->children[1] >= 0){
			
		int total = static_cast<int>(m_BVH.size()) - 1;
			
		//convert from m_BVH to bvhData index, as long as child is active!
		front = (bvh->children[0] != -1 && m_BVH[bvh->children[0]].isActive) ? total - bvh->children[0] : -1;
		back = (bvh->children[1] != -1 && m_BVH[bvh->children[1]].isActive) ? total - bvh->children[1] : -1;

		std::cout << "CHILDREN_IDX=" << bvh->children[0] << ":" << bvh->children[1] << std::endl;
	}

	int numTrisToAdd = static_cast<int>(bvh->triIdx.size());

	//if no tris and no children, skip
	//if (!bvh->isActive ||
	if(
		(front == -1 && back == -1 && numTrisToAdd == 0)) return 0;

	bvhData[bvhIdx] = BVH();
	bvhData[bvhIdx].radius = bvh->radius;
	bvhData[bvhIdx].centre[0] = bvh->centre[0];
	bvhData[bvhIdx].centre[1] = bvh->centre[1];
	bvhData[bvhIdx].centre[2] = bvh->centre[2];

	bvhData[bvhIdx].front = front;
	bvhData[bvhIdx].back = back;

	/*
	bvhData[bvhIdx].min[0] = bvh->min[0];
	bvhData[bvhIdx].min[1] = bvh->min[1];
	bvhData[bvhIdx].min[2] = bvh->min[2];

	bvhData[bvhIdx].max[0] = bvh->max[0];
	bvhData[bvhIdx].max[1] = bvh->max[1];
	bvhData[bvhIdx].max[2] = bvh->max[2];
	*/

	bvhData[bvhIdx].depth = bvh->depth;


	std::cout << "Tris to add:" << numTrisToAdd << std::endl;

	//NOTE: previously tried to add copies of the same BVH to account for this by "chunking" but not worth the complexity!
	//Currently, select a sensible value according to the triangle density of the scene and change BVH_CHUNK_SIZE
	if (numTrisToAdd > BVH_CHUNK_SIZE) throw new std::exception("Tried to add too many tris to BVH box");

	bvhData[bvhIdx].numTris = numTrisToAdd;
	bvhData[bvhIdx].hasTris = numTrisToAdd > 0;

	for (int t = 0; t < numTrisToAdd; t++) {
			
		//NOTE: x3 because BVH.triIdx is the index in verts, but BVH_BAKE (see: createBVH) has the literal triangle number
		bvhData[bvhIdx].triIdx[t] = bvh->triIdx[t]*3;
	}
	bvhBACKUP.push_back(bvhData[bvhIdx]);
	added++;
				
	return added;
}


//TODO: create bounding sphere BVH list and load into bvhData
//Traverse new bvh structure in raytracing code -> new structure + sphere bounds
int OBJLoader::createBVH(BVH* bvhData, int numBVH, Vertex* vertData, int numVerts) {

	int added = 0;
	//int lastAdded = -1;
	int totalBVH = static_cast<int>(m_BVH.size());

	for (int i = 0; i < totalBVH; i++) {

		int totalAdded = 0;
		
		totalAdded = putBVH(bvhData, &m_BVH[i], vertData, numVerts, added);

		//lastAdded = added;
		added += totalAdded;

	}

	if (added > numBVH) throw new std::exception("Tried to add more BVH than expected");

	std::cout << "Added:" << added << std::endl;

	return added;
}

void OBJLoader::freeBVHState() {
	m_BVH.clear();
}


///////////////////////////////////////////////////////////////////////////////
// OBJ loading code adapted from openGL/C++ examples, UEA
////////////////////////////////////////////////////////////////////////////////

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
		/*
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
		*/
		/*
		case 'u': //Change current Material
		{
			char buff[255];
			char buff2[255];
			fgets(line, 255, theFile); //read in the whole line	

			sscanf_s(line, "%s %s", &buff, sizeof(char) * 255, &buff2, sizeof(char) * 255);

			currentMat = lookupMaterial(buff2);

			break;
		}
		*/

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

	//04/05/19
	//Commented material import code
	//unused in the raytracer and causes compiler warnings with using %s in sscan_s
	//NOTE: materials will are a desirable feature for full OBJ support in the future; hence keeping code ready to go
	while (!fin.eof())

	{

		fin >> identifierStr;

		if (identifierStr == "#")

		{

			fin.getline(line, 255);

			fin >> std::ws;

		}
		
		/*
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
		*/

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

		/*
		else if (identifierStr == "map_Kd")

		{

			fin.getline(line, 255); //textureName

			fin >> std::ws;

			sscanf_s(line, "%s", &newMaterial.textureName, _countof(line));

		}
		*/
		/*
		else if (identifierStr == "map_Bump")//modified for normal map

		{

			fin.getline(line, 255); //textureName

			fin >> std::ws;

			newMaterial.bump = true;

			sscanf_s(line, "%s", &newMaterial.bumpTextureName, _countof(line));

		}
		*/
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

/////////////////////////////////////////////////////////////////////////////
//TRI BOX OVERLAP CODE ADAPTED FROM
//http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/
/////////////////////////////////////////////////////////////////////////////

//// START TRI BOX OVERLAP CODE

bool OBJLoader::planeBoxOverlap(vec3 &normal, vec3 &vert, vec3 &maxbox)
{

	int q;
	vec3 vmin, vmax;
	float v;
	for (q = 0; q <= 2; q++)
	{
		v = vert[q];
		if (normal[q] > 0.0f)
		{

			vmin[q] = -maxbox[q] - v;
			vmax[q] = maxbox[q] - v;
		}
		else
		{
			vmin[q] = maxbox[q] - v;
			vmax[q] = -maxbox[q] - v;
		}
	}

	if (dot2(normal, vmin) > 0.0f) return false;
	if (dot2(normal, vmax) >= 0.0f) return true;
	return false;

}


bool OBJLoader::SEPAXIS_X01(float a, float b, float fa, float fb, vec3 &v0, vec3 &v2, vec3 &boxhalfsize) {
	float min1, max1 = 0.0f;
	float p0 = a * v0[1] - b * v0[2];
	float p2 = a * v2[1] - b * v2[2];
	if (p0 < p2) { min1 = p0; max1 = p2; }
	else { min1 = p2; max1 = p0; }
	float rad = fa * boxhalfsize[1] + fb * boxhalfsize[2];
	return (min1 > rad || max1 < -rad);
}
bool OBJLoader::SEPAXIS_X2(float a, float b, float fa, float fb, vec3 &v0, vec3 &v1, vec3 &boxhalfsize) {
	float min1, max1 = 0.0f;
	float p0 = a * v0[1] - b * v0[2];
	float p1 = a * v1[1] - b * v1[2];
	if (p0 < p1) { min1 = p0; max1 = p1; }
	else { min1 = p1; max1 = p0; }
	float rad = fa * boxhalfsize[1] + fb * boxhalfsize[2];
	return (min1 > rad || max1 < -rad);
}
bool OBJLoader::SEPAXIS_Y02(float a, float b, float fa, float fb, vec3 &v0, vec3 &v2, vec3 &boxhalfsize) {
	float min1, max1 = 0.0f;
	float p0 = -a * v0[0] + b * v0[2];
	float p2 = -a * v2[0] + b * v2[2];
	if (p0 < p2) { min1 = p0; max1 = p2; }
	else { min1 = p2; max1 = p0; }
	float rad = fa * boxhalfsize[0] + fb * boxhalfsize[2];
	return (min1 > rad || max1 < -rad);
}
bool OBJLoader::SEPAXIS_Y1(float a, float b, float fa, float fb, vec3 &v0, vec3 &v1, vec3 &boxhalfsize) {
	float min1, max1 = 0.0f;
	float p0 = -a * v0[0] + b * v0[2];
	float p1 = -a * v1[0] + b * v1[2];
	if (p0 < p1) { min1 = p0; max1 = p1; }
	else { min1 = p1; max1 = p0; }
	float rad = fa * boxhalfsize[0] + fb * boxhalfsize[2];
	return (min1 > rad || max1 < -rad);
}
bool OBJLoader::SEPAXIS_Z12(float a, float b, float fa, float fb, vec3 &v1, vec3 &v2, vec3 &boxhalfsize) {
	float min1, max1 = 0.0f;
	float p1 = a * v1[0] - b * v1[1];
	float p2 = a * v2[0] - b * v2[1];
	if (p2 < p1) { min1 = p2; max1 = p1; }
	else { min1 = p1; max1 = p2; }
	float rad = fa * boxhalfsize[0] + fb * boxhalfsize[1];
	return (min1 > rad || max1 < -rad);
}
bool OBJLoader::SEPAXIS_Z0(float a, float b, float fa, float fb, vec3 &v0, vec3 &v1, vec3 &boxhalfsize) {
	float min1, max1 = 0.0f;
	float p0 = a * v0[0] - b * v0[1];
	float p1 = a * v1[0] - b * v1[1];
	if (p0 < p1) { min1 = p0; max1 = p1; }
	else { min1 = p1; max1 = p0; }
	float rad = fa * boxhalfsize[0] + fb * boxhalfsize[1];
	return (min1 > rad || max1 < -rad);
}


bool OBJLoader::triBoxOverlap(vec3 &boxcenter, vec3 &boxhalfsize, vec4 &vert1, vec4 &vert2, vec4 &vert3, vec3 &normal)
{

	//verts relative to box
	vec3 v0 = { 0 };
	v0[0] = vert1[0] - boxcenter[0];
	v0[1] = vert1[1] - boxcenter[1];
	v0[2] = vert1[2] - boxcenter[2];
	vec3 v1 = { 0 };
	v1[0] = vert2[0] - boxcenter[0];
	v1[1] = vert2[1] - boxcenter[1];
	v1[2] = vert2[2] - boxcenter[2];
	vec3 v2 = { 0 };
	v2[0] = vert3[0] - boxcenter[0];
	v2[1] = vert3[1] - boxcenter[1];
	v2[2] = vert3[2] - boxcenter[2];

	//edges
	vec3 e0 = { 0 };
	e0[0] = v1[0] - v0[0];
	e0[1] = v1[1] - v0[1];
	e0[2] = v1[2] - v0[2];
	vec3 e1 = { 0 };
	e1[0] = v2[0] - v1[0];
	e1[1] = v2[1] - v1[1];
	e1[2] = v2[2] - v1[2];
	vec3 e2 = { 0 };
	e2[0] = v0[0] - v2[0];
	e2[1] = v0[1] - v2[1];
	e2[2] = v0[2] - v2[2];

	//these 9 tests first is "faster"
	float fex = abs(e0[0]);
	float fey = abs(e0[1]);
	float fez = abs(e0[2]);
	if (SEPAXIS_X01(e0[2], e0[1], fez, fey, v0, v2, boxhalfsize)
		|| SEPAXIS_Y02(e0[2], e0[0], fez, fex, v0, v2, boxhalfsize)
		|| SEPAXIS_Z12(e0[1], e0[0], fey, fex, v1, v2, boxhalfsize)) return false;

	fex = abs(e1[0]);
	fey = abs(e1[1]);
	fez = abs(e1[2]);
	if (SEPAXIS_X01(e1[2], e1[1], fez, fey, v0, v2, boxhalfsize)
		|| SEPAXIS_Y02(e1[2], e1[0], fez, fex, v0, v2, boxhalfsize)
		|| SEPAXIS_Z0(e1[1], e1[0], fey, fex, v0, v1, boxhalfsize)) return false;

	fex = abs(e2[0]);
	fey = abs(e2[1]);
	fez = abs(e2[2]);
	if (SEPAXIS_X2(e2[2], e2[1], fez, fey, v0, v1, boxhalfsize)
		|| SEPAXIS_Y1(e2[2], e2[0], fez, fex, v0, v1, boxhalfsize)
		|| SEPAXIS_Z12(e2[1], e2[0], fey, fex, v1, v2, boxhalfsize)) return false;


	//first test overlap in the {x,y,z}-directions
	//find min, max of the triangle each direction, and test for overlap in
	//that direction -- this is equivalent to testing a minimal AABB around
	//the triangle against the AABB

	//In X direction
	float min1 = min(v0[0], min(v1[0], v2[0]));
	float max1 = max(v0[0], max(v1[0], v2[0]));
	if (min1 > boxhalfsize[0] || max1 < -boxhalfsize[0]) return false;

	//In Y direction
	min1 = min(v0[1], min(v1[1], v2[1]));
	max1 = max(v0[1], max(v1[1], v2[1]));
	if (min1 > boxhalfsize[1] || max1 < -boxhalfsize[1]) return false;

	//In Z direction
	min1 = min(v0[2], min(v1[2], v2[2]));
	max1 = max(v0[2], max(v1[2], v2[2]));
	if (min1 > boxhalfsize[2] || max1 < -boxhalfsize[2]) return false;

	//test if the box intersects the plane of the triangle
	//compute plane equation of triangle: normal*x+d=0

	//vec3 normal = cross(e0,e1);

	if (!planeBoxOverlap(normal, v0, boxhalfsize)) return false;

	return true;   //overlap!
}

//// END TRI BOX OVERLAP CODE
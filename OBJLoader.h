#pragma once

#include <stdio.h>
#include <vector>
#include "Vertex.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>

#define MAX_BVH_DEPTH 2

struct ObjMat
{
	//Loaded from the File
	char matName[255];
	float ambientCol[3];
	float diffuseCol[3];
	float specCol[3];
	float d;
	float ns;
	int illum;
	char textureName[255] = { 0 };
	char bumpTextureName[255] = { 0 };
	bool bump = false;

	//Created
	int glIndex = 0;
	int glIndexBump = 0;
};

//-------------------------------------------------------------
//- SObjFace
//- Indices for each face of the OBJ file
class SObjFace
{
public:

	unsigned int m_uiVertIdx[3];
	unsigned int m_uiNormalIdx[3];
	unsigned int m_uiTexCoordIdx[3];

	int matId;

	SObjFace(unsigned int vert[3], unsigned int text[3], unsigned int norm[3], int newMatId)
	{
		//copy the data;
		m_uiVertIdx[0] = vert[0] - 1;
		m_uiVertIdx[1] = vert[1] - 1;
		m_uiVertIdx[2] = vert[2] - 1;

		m_uiTexCoordIdx[0] = text[0] - 1;
		m_uiTexCoordIdx[1] = text[1] - 1;
		m_uiTexCoordIdx[2] = text[2] - 1;

		m_uiNormalIdx[0] = norm[0] - 1;
		m_uiNormalIdx[1] = norm[1] - 1;
		m_uiNormalIdx[2] = norm[2] - 1;

		matId = newMatId;
	}

	bool operator<(const SObjFace &face2) const  //Defined to sort list
	{
		if (matId < face2.matId) return true;
		else return false;
	}

};

//Wrapper for vec3 since you cannot store an array directly in an STL vector object
struct vector3d {

	vec3 pos = { 0 };

};


class OBJLoader {

public:
	
	static int loadRawVertexList(const char * fileName, Vertex** vertData);
	static void loadVertices(Vertex* vertData, int numVerts);
	static int countBVHNeeded(Vertex* vertData, int numVerts);
	static int createBVH(BVH* bvhData, int numBVH, Vertex* vertData, int numVerts);
	

private:
	

	class TriangleBounds {
	public:
		unsigned int mortonCode;
		int objId;
	};
	static bool OBJLoader::mortonCodeSort(TriangleBounds a, TriangleBounds b);

	static BVH_BAKE*  OBJLoader::generateHierarchy(unsigned int* sortedMortonCodes,
		int*          sortedObjectIDs,
		int           first,
		int           last,
		float* radii,
		int numObj);
	static void OBJLoader::createLinearBVH(std::vector<vector3d> barycentres, std::vector<float> radii, float* sceneCentroid, float* cubeSize);

	static int putBVH(BVH* bvhData, BVH_BAKE* bvh, Vertex* vertData, int numVerts, int added, int depth);
	static int putVertsInBVH(Vertex* vertData, int numVerts, BVH_BAKE* bvh, int depth);

	static void readTriangleFaceVertTexNorm(char *line, int matId);
	static void readFaceLine(FILE * theFile, int matId);
	static bool myFileLoader(const char *filename);
	static bool myMTLLoader(const char *mainName, const char *filename);
	static int lookupMaterial(char *matName);
	static void splitFrontString(char * inputString, char * frontString, char * restString, int size);

	static std::vector<BVH_BAKE> m_BVH;

	static std::vector<Vertex> m_distinctVerts;
	static std::vector<vector3d> m_vVertices;
	//an STL vector to hold vertex normals
	static std::vector<vector3d> m_vNormals;
	//an STL vector to hold the texture coordinates
	static std::vector<vector3d> m_vTexCoords;
	//an STL vector for the faces
	static std::vector<SObjFace> m_vFaces;
	static std::vector<ObjMat> theMats;


	/////////////////////////////////////////////////////////////////////////////
	//TRI BOX OVERLAP CODE ADAPTED FROM
	//http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/
	/////////////////////////////////////////////////////////////////////////////
	static bool planeBoxOverlap(vec3 &normal, vec3 &vert, vec3 &maxbox);
	static bool SEPAXIS_X01(float a, float b, float fa, float fb, vec3 &v0, vec3 &v2, vec3 &boxhalfsize);
	static bool SEPAXIS_X2(float a, float b, float fa, float fb, vec3 &v0, vec3 &v1, vec3 &boxhalfsize);
	static bool SEPAXIS_Y02(float a, float b, float fa, float fb, vec3 &v0, vec3 &v2, vec3 &boxhalfsize);
	static bool SEPAXIS_Y1(float a, float b, float fa, float fb, vec3 &v0, vec3 &v1, vec3 &boxhalfsize);
	static bool SEPAXIS_Z12(float a, float b, float fa, float fb, vec3 &v1, vec3 &v2, vec3 &boxhalfsize);
	static bool SEPAXIS_Z0(float a, float b, float fa, float fb, vec3 &v0, vec3 &v1, vec3 &boxhalfsize);
	static bool triBoxOverlap(vec3 &boxcenter, vec3 &boxhalfsize, vec4 &vert1, vec4 &vert2, vec4 &vert3, vec3 &normal);

};
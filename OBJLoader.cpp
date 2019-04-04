#include "OBJLoader.h"


std::vector<vector3d> OBJLoader::m_vVertices = std::vector<vector3d>();
std::vector<vector3d> OBJLoader::m_vNormals = std::vector<vector3d>();
std::vector<vector3d> OBJLoader::m_vTexCoords = std::vector<vector3d>();
std::vector<SObjFace> OBJLoader::m_vFaces = std::vector<SObjFace>();
std::vector<ObjMat> OBJLoader::theMats = std::vector<ObjMat>();

int OBJLoader::loadRawVertexList(const char * fileName, Vertex** vertData, float scale) {

	if (!OBJLoader::myFileLoader(fileName)){
		throw std::exception("OBJLoader: Failed to load model.");
	}

	std::vector<Vertex> distinctVerts = std::vector<Vertex>();

	//loop over faces, store triangle verts
	for (int face = 0; face < m_vFaces.size(); face++) {

		int vertAIdx = m_vFaces[face].m_uiVertIdx[0];
		int vertBIdx = m_vFaces[face].m_uiVertIdx[1];
		int vertCIdx = m_vFaces[face].m_uiVertIdx[2];

		int normAIdx = m_vFaces[face].m_uiNormalIdx[0];
		int normBIdx = m_vFaces[face].m_uiNormalIdx[1];
		int normCIdx = m_vFaces[face].m_uiNormalIdx[2];

		//TODO: use vertex indexing
		//Get distinct vertex points
		Vertex v1 = Vertex();
		v1.pos[0] = m_vVertices[vertAIdx].pos[0];
		v1.pos[1] = m_vVertices[vertAIdx].pos[1];
		v1.pos[2] = m_vVertices[vertAIdx].pos[2];
		v1.color[0] = abs(m_vNormals[normAIdx].pos[0]);
		v1.color[1] = abs(m_vNormals[normAIdx].pos[1]);
		v1.color[2] = abs(m_vNormals[normAIdx].pos[2]);

		Vertex v2 = Vertex();
		v2.pos[0] = m_vVertices[vertBIdx].pos[0];
		v2.pos[1] = m_vVertices[vertBIdx].pos[1];
		v2.pos[2] = m_vVertices[vertBIdx].pos[2];
		v2.color[0] = abs(m_vNormals[normBIdx].pos[0]);
		v2.color[1] = abs(m_vNormals[normBIdx].pos[1]);
		v2.color[2] = abs(m_vNormals[normBIdx].pos[2]);

		Vertex v3 = Vertex();
		v3.pos[0] = m_vVertices[vertCIdx].pos[0];
		v3.pos[1] = m_vVertices[vertCIdx].pos[1];
		v3.pos[2] = m_vVertices[vertCIdx].pos[2];
		v3.color[0] = abs(m_vNormals[normCIdx].pos[0]);
		v3.color[1] = abs(m_vNormals[normCIdx].pos[1]);
		v3.color[2] = abs(m_vNormals[normCIdx].pos[2]);

		distinctVerts.push_back(v1);
		distinctVerts.push_back(v2);
		distinctVerts.push_back(v3);

	}

	int size = distinctVerts.size();
	std::cout << "Loading " << size << " vertices (scale=" << scale * 100.0f << "%) from " << fileName << std::endl;

	//copy list into vertData
	*vertData = (Vertex*)malloc(size * sizeof(Vertex));

	for (int v = 0; v < size; v++) {
		*vertData[v] = Vertex(distinctVerts[v]);
	}

	//return size
	return size;
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
				//Convert coordinates from 3DS Max coordinate system to OpenGL
				thePoints[0] *= -1.0f;
				thePoints[2] *= -1.0f;


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
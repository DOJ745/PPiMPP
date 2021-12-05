using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChunkPlacer : MonoBehaviour
{
    [Header("Settings")]
    public Transform ship;
    public Chunk[] chunkPrefabs;
    public Chunk firstChunk;

    private List<Chunk> spawnedChunks = new List<Chunk>();
    // Start is called before the first frame update
    void Start()
    {
        spawnedChunks.Add(firstChunk);
    }

    // Update is called once per frame
    void Update()
    {
        if(ship.position.z > spawnedChunks[spawnedChunks.Count - 1].end.position.z - 100)
        {
            SpawnChunks();
        }
    }

    private void SpawnChunks()
    {
        Chunk newChunk = Instantiate(chunkPrefabs[Random.Range(0, chunkPrefabs.Length)]);
        newChunk.transform.position = spawnedChunks[spawnedChunks.Count - 1].end.position - newChunk.begin.localPosition;

        spawnedChunks.Add(newChunk);

        if (spawnedChunks.Count >= 15)
        {
            Destroy(spawnedChunks[0].gameObject);
            spawnedChunks.RemoveAt(0);
        }
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AsteroidSpawner : MonoBehaviour
{
    [Header("Settings")]
    public GameObject asteroidPrefab;
    public float respawnTime = 5.0f;
    public float maxRespawnTime = 10.0f;
    public float minRespawnTime = 3.0f;

    void Start()
    {
        StartCoroutine(asteroidWave());
        InvokeRepeating("increaseDifficulty", 5f, 15f);
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void spawnAsteroids()
    {
        GameObject asteroid = Instantiate(asteroidPrefab);

        asteroid.transform.position = new Vector3(
            Random.Range(Camera.main.transform.localPosition.x - 1.85f, Camera.main.transform.localPosition.x + 3.5f),
            Camera.main.transform.localPosition.y - 0.85f,
            Camera.main.transform.localPosition.z + 100.0f); 
    }

    private void increaseDifficulty()
    {
        respawnTime = (respawnTime <= maxRespawnTime && respawnTime > maxRespawnTime) ? respawnTime-- : minRespawnTime;
    }
    private IEnumerator asteroidWave()
    {
        while (true)
        {
            Debug.Log("Started at timestamp : " + Time.time);
            yield return new WaitForSeconds(respawnTime);
            Debug.Log("Finished at timestamp : " + Time.time);
            spawnAsteroids();
        }
    }
}

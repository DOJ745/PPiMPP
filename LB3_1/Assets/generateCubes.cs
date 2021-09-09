using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class generateCubes : MonoBehaviour
{
    // Start is called before the first frame update
    public GameObject prefabOne;
    public GameObject prefabTwo;

    void Start()
    {
        Vector3 positionOne = new Vector3(Random.Range(-9.0f, 9.0f), 0.2f, Random.Range(-13.9f, 10.0f));
        Vector3 positionTwo = new Vector3(Random.Range(-9.1f, 9.1f), 0.5f, Random.Range(-12.0f, 10.0f));

        for(int i = 0; i < 3; i++) {
            Instantiate(prefabOne, positionOne, Quaternion.identity);
            Instantiate(prefabTwo, positionTwo, Quaternion.identity);
        }
    }

    // Update is called once per frame
    void Update()
    {

    }
}

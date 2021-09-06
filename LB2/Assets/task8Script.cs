using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class task8Script : MonoBehaviour
{
    private MeshRenderer render;
    private float minX;
    private float minZ;

    private float maxX;
    private float maxZ;
    // Start is called before the first frame update
    void Start()
    {
        render = GetComponent<MeshRenderer>();

        minX = render.bounds.min.x;
        minZ = render.bounds.min.z;

        maxX = render.bounds.max.x;
        maxZ = render.bounds.max.z;
    }

    // Update is called once per frame
    void Update()
    {
        float newX = Random.Range(minX, maxX);
        float newZ = Random.Range(minZ, maxZ);
        float newY = transform.position.y + 5;

        float rndRed = Random.Range(0.0f, 1.0f);
        float rndGreen = Random.Range(0.0f, 1.0f);
        float rndBlue = Random.Range(0.0f, 1.0f);

        Color randomColor = new Color(rndRed, rndGreen, rndBlue, 1.0f);

        if(Input.GetKeyDown(KeyCode.Space)) {
            GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.transform.position = new Vector3(newX, newY, newZ);

            cube.AddComponent<Rigidbody>();
        }

        if(Input.GetKeyDown(KeyCode.Mouse0)){

            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.transform.position = new Vector3(newX, newY, newZ);

            Renderer sphereRenderer = sphere.GetComponent<Renderer>();
            sphereRenderer.material.color = randomColor;

            sphere.AddComponent<Rigidbody>();
        }
    }
}

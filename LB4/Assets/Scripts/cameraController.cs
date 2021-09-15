using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class cameraController : MonoBehaviour
{
    // Start is called before the first frame update

    public GameObject obj;
    private Vector3 offset;

    void Start()
    {
        offset = transform.position - obj.transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        transform.position = obj.transform.position + offset;
    }
}

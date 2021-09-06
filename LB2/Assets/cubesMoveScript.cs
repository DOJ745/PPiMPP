using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class cubesMoveScript : MonoBehaviour
{
    public GameObject obj;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        transform.position = Vector3.MoveTowards(transform.position, obj.transform.position, Time.deltaTime);
    }
}

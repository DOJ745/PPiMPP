using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class someForce : MonoBehaviour
{
    public float xForce;
    public float yForce;
    public float zForce;

    private void OnCollisionEnter(Collision otherObject) {
        if(
            otherObject.gameObject.name == "Cube (2)" ||
            otherObject.gameObject.name == "Cube (1)" ||
            otherObject.gameObject.name == "Cube"
            ) 
        {
            Debug.Log("Hit the cube");
            otherObject.rigidbody.AddForce(xForce, yForce, zForce);
        }
    }
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }
}

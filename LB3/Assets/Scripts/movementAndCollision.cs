using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class movementAndCollision : MonoBehaviour
{
    // Start is called before the first frame update
    public float speed = 5;
    private Rigidbody rigidBody;

    private void OnCollisionEnter(Collision myObject) {

        if( myObject.gameObject.name == "GreenCube" || myObject.gameObject.name == "GreenCube(Clone)") 
        {
            Debug.Log("Hit the green cube");
            Destroy(myObject.gameObject, 0.0f);
        }

        if( myObject.gameObject.name == "RedCube" || myObject.gameObject.name == "RedCube(Clone)" ) 
        {
            Debug.Log("Hit the red cube");
            Destroy(gameObject, 0.5f);
        }
    }

    void Start()
    {
        rigidBody = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        float moveHorizontal = Input.GetAxis("Horizontal");
        float moveVertical = Input.GetAxis("Vertical");

        Vector3 movement = new Vector3(moveHorizontal, 0.0f, moveVertical);
        rigidBody.AddForce(movement * speed);
    }
}

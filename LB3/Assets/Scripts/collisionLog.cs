using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class collisionLog : MonoBehaviour
{
    private void OnCollisionEnter(Collision myCollision) {
        if(myCollision.gameObject.name == "Floor"){
            Debug.Log("Hit the floor");
        }
        if(myCollision.gameObject.name == "Wall"){
            Debug.Log("Hit the wall");
        }
        if(myCollision.gameObject.name == "Wall (1)"){
            Debug.Log("Hit the wall");
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

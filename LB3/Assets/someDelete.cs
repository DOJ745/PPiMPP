using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class someDelete : MonoBehaviour
{
    private void OnCollisionEnter(Collision otherObject) {
        if(
            otherObject.gameObject.name == "Cube (2)" ||
            otherObject.gameObject.name == "Cube (1)" ||
            otherObject.gameObject.name == "Cube"
            ) 
        {
            Debug.Log("Hit the cube");
            Destroy(otherObject.gameObject, 0.25f);
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

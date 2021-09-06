using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class changeColorScript : MonoBehaviour
{
    private Renderer renderer;
    // Start is called before the first frame update
    void Start()
    {
        renderer = GetComponent<Renderer>();
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown("space")) {
            renderer.material.color = Color.red;
        }
        
    }
}

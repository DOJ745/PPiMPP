using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Kill : MonoBehaviour
{
    // Start is called before the first frame update
    public Camera currentCamera;
    void Start()
    {
        currentCamera = this.GetComponent<Camera>();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = currentCamera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit))
            {
                if(hit.collider.gameObject.name == "Enemy" || hit.collider.gameObject.name == "EnemySphere")
                {
                    Destroy(hit.collider.gameObject);
                }
            }
        }
    }
}

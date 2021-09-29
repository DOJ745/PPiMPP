using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class pushCube : MonoBehaviour
{
    // Start is called before the first frame update
    public Camera current_camera;
    //public Rigidbody rigidBody;
    //public float force;

    public float xForce = 1.0f;
    public float yForce = 1.0f;
    public float zForce = 1.0f;

    void Start()
    {
   
    }

    // Update is called once per frame
    void Update()
    {
        float rndRed = Random.Range(0.0f, 1.0f);
        float rndGreen = Random.Range(0.0f, 1.0f);
        float rndBlue = Random.Range(0.0f, 1.0f);

        Color randomColor = new Color(rndRed, rndGreen, rndBlue, 1.0f);

        Vector3 apllyingForce = new Vector3(xForce, yForce, zForce);

        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = current_camera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit))
            {
                Renderer current_renderer = hit.collider.GetComponent<Renderer>();
                hit.rigidbody.AddForce(apllyingForce, ForceMode.Impulse);
                current_renderer.material.color = randomColor;
                //agent.SetDestination(hit.point);
                //rigidBody.AddForce(apllyingForce, ForceMode.Impulse);
            }
        }
        
    }
}

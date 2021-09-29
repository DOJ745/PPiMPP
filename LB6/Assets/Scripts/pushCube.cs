using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PushCube : MonoBehaviour
{

    private float inputXForce = 1.0f;
    private float inputYForce = 1.0f;
    private float inputZForce = 0.0f;

    public Camera current_camera;

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

        Vector3 apllyingForce = new Vector3(inputXForce, inputYForce, inputZForce);

        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = current_camera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit))
            {
                Renderer current_renderer = hit.collider.GetComponent<Renderer>();
                hit.rigidbody.AddForce(apllyingForce, ForceMode.Impulse);

                if(current_renderer != null)
                    current_renderer.material.color = randomColor;
            }
        }
        
    }

    public void ReadXInput(string input)
    {
        Debug.Log("Input for X - " + input);
        inputXForce = float.Parse(input);
    }

    public void ReadYInput(string input)
    {
        Debug.Log("Input for Y - " + input);
        inputYForce = float.Parse(input);
    }

    public void ReadZInput(string input)
    {
        Debug.Log("Input for Z - " + input);
        inputZForce = float.Parse(input);
    }
}

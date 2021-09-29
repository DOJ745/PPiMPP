using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FlyCamera : MonoBehaviour
{

    /*
    Writen by Windexglow 11-13-10.  Use it, edit it, steal it I don't care.  
    Converted to C# 27-02-13 - no credit wanted.
    Simple flycam I made, since I couldn't find any others made public.  
    Made simple to use (drag and drop, done) for regular keyboard layout  
    wasd : basic movement
    shift : Makes camera accelerate
    space : Moves camera on X and Z axis only.  So camera doesn't gain any height*/

    // Regular speed
    public float mainSpeed = 100.0f;

    // Multiplied by how long shift is held.  Basically running
    public float shiftAdd = 250.0f; 

    // Maximum speed when holdin gshift
    public float maxShift = 1000.0f; 

    // How sensitive it with mouse
    public float camSens = 0.25f;

    // Kind of in the middle of the screen, rather than at the top (play)
    private Vector3 lastMouse = new Vector3(0, 5.5f, -11);
    
    private float totalRun = 1.0f;

    void Update()
    {
        lastMouse = Input.mousePosition - lastMouse;
        lastMouse = new Vector3(-lastMouse.y * camSens, lastMouse.x * camSens, 0);

        lastMouse = new Vector3(transform.eulerAngles.x + lastMouse.x, 
            transform.eulerAngles.y + lastMouse.y, 0);

        transform.eulerAngles = lastMouse;
        lastMouse = Input.mousePosition;
        //Mouse  camera angle done.  

        // Keyboard commands
        float f = 0.0f;
        Vector3 p = GetBaseInput();
        if (p.sqrMagnitude > 0)
        { 
            // Only move while a direction key is pressed
            if (Input.GetKey(KeyCode.LeftShift))
            {
                totalRun += Time.deltaTime;
                p = p * totalRun * shiftAdd;
                p.x = Mathf.Clamp(p.x, -maxShift, maxShift);
                p.y = Mathf.Clamp(p.y, -maxShift, maxShift);
                p.z = Mathf.Clamp(p.z, -maxShift, maxShift);
            }
            else
            {
                totalRun = Mathf.Clamp(totalRun * 0.5f, 1f, 1000f);
                p = p * mainSpeed;
            }

            p = p * Time.deltaTime;
            Vector3 newPosition = transform.position;
            if (Input.GetKey(KeyCode.Space))
            { 
                //If player wants to move on X and Z axis only
                transform.Translate(p);
                newPosition.x = transform.position.x;
                newPosition.z = transform.position.z;
                transform.position = newPosition;
            }
            else
            {
                transform.Translate(p);
            }
        }
    }

    private Vector3 GetBaseInput()
    { 
        // Returns the basic values, if it's 0 than it's not active.
        Vector3 apply_Velocity = new Vector3();
        if (Input.GetKey(KeyCode.W))
        {
            apply_Velocity += new Vector3(0, 0, 1);
        }
        if (Input.GetKey(KeyCode.S))
        {
            apply_Velocity += new Vector3(0, 0, -1);
        }
        if (Input.GetKey(KeyCode.A))
        {
            apply_Velocity += new Vector3(-1, 0, 0);
        }
        if (Input.GetKey(KeyCode.D))
        {
            apply_Velocity += new Vector3(1, 0, 0);
        }
        return apply_Velocity;
    }
}

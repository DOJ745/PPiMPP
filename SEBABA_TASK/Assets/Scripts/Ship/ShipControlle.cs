using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShipControlle : MonoBehaviour
{
    [Header("Settings")]
    public float speed = 5f;
    public float rotationSpeed = 50f;

    [Header("Limit Rotation")]
    public float maxRotate = 40f;
    public float minRotate = -40f;

    private Rigidbody rigidBody;
    private Transform currentTransform;

    //private Quaternion target;
    // Start is called before the first frame update
    void Start()
    {
        rigidBody = GetComponent<Rigidbody>();
        currentTransform = GetComponent<Transform>();
    }

    // Update is called once per frame
    void Update()
    {
        float moveHorizontal = Input.GetAxis("Horizontal");
        float moveVertical = Input.GetAxis("Vertical");

        Vector3 movement = new Vector3(moveHorizontal, 0.0f, moveVertical);

        Debug.Log("Z ANGLE - " + currentTransform.eulerAngles.z);

        if (Input.GetKey(KeyCode.A))
        {
            currentTransform.Rotate(Vector3.forward * rotationSpeed * Time.deltaTime);
        }

        if (Input.GetKey(KeyCode.D))
        {
            currentTransform.Rotate(Vector3.forward * -1 * rotationSpeed * Time.deltaTime);
        }

        LimitRotation();
        rigidBody.AddForce(movement * speed);
    }

    private void LimitRotation()
    {
        Vector3 targetEulerAngles = currentTransform.rotation.eulerAngles;

        targetEulerAngles.z = (targetEulerAngles.z > 180) ? targetEulerAngles.z - 360 : targetEulerAngles.z;
        targetEulerAngles.z = Mathf.Clamp(targetEulerAngles.z, minRotate, maxRotate);

        currentTransform.rotation = Quaternion.Euler(targetEulerAngles);
    }
}

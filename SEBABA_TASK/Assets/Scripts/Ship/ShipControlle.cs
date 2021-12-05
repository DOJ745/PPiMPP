using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShipControlle : MonoBehaviour
{
    [Header("Settings")]
    public float speed = 5f;
    public float rotationSpeed = 50f;

    [Header("Limit Speed")]
    public float maxSpeed = 10f;
    public float minSpeed = 5f;

    [Header("Limit Rotation")]
    public float maxRotate = 40f;
    public float minRotate = -40f;

    [Header("Score data")]
    public GameObject gameDataManagerObj;

    private Rigidbody rigidBody;
    private Transform currentTransform;

    private GameDataManager gameDataManager;


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

        if (Input.GetKey(KeyCode.A))
        {
            currentTransform.Rotate(Vector3.forward * rotationSpeed * Time.deltaTime);
        }

        if (Input.GetKeyUp(KeyCode.A) && currentTransform.rotation.eulerAngles.z != 0 )
        {
            SetDefRotation();
        }

        if (Input.GetKey(KeyCode.D))
        {
            currentTransform.Rotate(Vector3.forward * -1 * rotationSpeed * Time.deltaTime);
        }

        if (Input.GetKeyUp(KeyCode.D) && currentTransform.rotation.eulerAngles.z != 0)
        {
            SetDefRotation();
        }

        if (Input.GetKey(KeyCode.Space))
        {
            speed += 0.15f;
        }

        LimitSpeed();
        LimitRotation();
        rigidBody.AddForce(movement * speed);
    }

    private void SetDefRotation()
    {
        currentTransform.rotation = Quaternion.Euler(
                currentTransform.rotation.eulerAngles.x,
                currentTransform.rotation.eulerAngles.y,
                0);
    }

    private void LimitRotation()
    {
        Vector3 targetEulerAngles = currentTransform.rotation.eulerAngles;

        targetEulerAngles.z = (targetEulerAngles.z > 180) ? targetEulerAngles.z - 360 : targetEulerAngles.z;
        targetEulerAngles.z = Mathf.Clamp(targetEulerAngles.z, minRotate, maxRotate);

        currentTransform.rotation = Quaternion.Euler(targetEulerAngles);
    }

    private void LimitSpeed()
    {
        speed = (speed < maxSpeed) ? speed : maxSpeed;
        speed = Mathf.Clamp(speed, minSpeed, maxSpeed);
    }

}

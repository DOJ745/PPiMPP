using UnityEngine;

public class SmoothFollow : MonoBehaviour
{
    [Header("Settings")]
    public float distance = 10.0f;
    public float height = 5.0f;
    public float heightDamping = 2.0f;
    public float rotationDamping = 3.0f;
    [Header("Zoom")]
    public float zoomSpeedIn = 1.15f;
    [Header("Limitations")]
    public float minDistance = 10.0f;
    public float defaultDistance = 10.0f;
    public Transform target;

    private void LateUpdate()
    {
        // Early out if we don't have a target
        if (!target)
        {
            return;
        }

        // Calculate the current rotation angles
        float wantedRotationAngle = target.eulerAngles.y;
        float wantedHeight = target.position.y + height;

        float currentRotationAngle = transform.eulerAngles.y;
        float currentHeight = transform.position.y;

        // Damp the rotation around the y-axis
        currentRotationAngle = Mathf.LerpAngle(
            currentRotationAngle, 
            wantedRotationAngle, 
            rotationDamping * Time.deltaTime);

        // Damp the height
        currentHeight = Mathf.Lerp(currentHeight, wantedHeight, heightDamping * Time.deltaTime);

        // Convert the angle into a rotation
        Quaternion currentRotation = Quaternion.Euler(0, currentRotationAngle, 0);

        // Set the position of the camera on the x-z plane to:
        // distance meters behind the target
        var pos = transform.position;
        pos = target.position - currentRotation * Vector3.forward * distance;
        pos.y = currentHeight;
        transform.position = pos;

        // Always look at the target
        transform.LookAt(target);

        if (Input.GetKey(KeyCode.Space))
        {
            distance = (distance > minDistance) ? (distance -= zoomSpeedIn * Time.deltaTime): minDistance;
        }
        if (Input.GetKeyUp(KeyCode.Space))
        {
            distance = defaultDistance;
        }
    }
}
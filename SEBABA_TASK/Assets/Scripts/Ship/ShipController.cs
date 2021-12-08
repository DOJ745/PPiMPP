using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class ShipController : MonoBehaviour
{
    [Header("Settings")]
    public float speed = 5f;
    public float turnSpeed = 5f;
    public float rotationSpeed = 50f;

    [Header("Limit Speed")]
    public float maxSpeed = 10f;
    public float minSpeed = 5f;

    [Header("Limit Rotation")]
    public float maxRotate = 40f;
    public float minRotate = -40f;

    private Rigidbody rigidBody;
    private Transform currentTransform;

    private Timer scoreData;

    private void OnCollisionEnter(Collision myCollision)
    {
        if (myCollision.gameObject.name.Contains("Asteroid"))
        {
            Debug.Log("Hit the Asteroid");

            GameObject.Find("Sound Manager").GetComponent<AudioSource>().volume = 0;

            transform.localScale = new Vector3(0f, 0f, 0f);

            scoreData = Camera.main.GetComponent<Timer>();
            GameObject gameDataManagerObj = scoreData.gameDataManagerObj;

            GameDataManager gameDataManager = gameDataManagerObj.GetComponent<GameDataManager>();

            Scores currentScores = gameDataManager.readScores();
            currentScores.addScore(scoreData.currentScore);
            gameDataManager.writeFile(currentScores);

            destroyObjects();
            StartCoroutine(holdLoading());
        }
    }
    private IEnumerator holdLoading()
    {

        Debug.Log("Started EXITING AT: " + Time.time);
        yield return new WaitForSeconds(3);

        SceneManager.LoadScene("MainMenu");
        Debug.Log("Finished EXITING AT: " + Time.time);
    }


    // Start is called before the first frame update
    void Start()
    {
        rigidBody = GetComponent<Rigidbody>();
        currentTransform = GetComponent<Transform>();
        //SoundManager.PlaySound("background");
    }

    // Update is called once per frame
    void Update()
    {
        float moveHorizontal = Input.GetAxis("Horizontal");
        currentTransform.rotation = Quaternion.Euler(0, 0, currentTransform.rotation.eulerAngles.z);

        Vector3 movement = new Vector3(moveHorizontal * turnSpeed, 0.0f, speed);

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

        if (Input.GetKeyUp(KeyCode.Space))
        {
            speed = minSpeed;
        }

        if (Input.GetKey(KeyCode.Escape))
        {
            Application.Quit();
        }

        LimitSpeed();
        LimitRotation();

        if(rigidBody != null) { rigidBody.AddForce(movement * speed); }
    }

    private void SetDefRotation()
    {
        currentTransform.rotation = Quaternion.Euler(0, 0, 0);
    }

    private void LimitRotation()
    {
        Vector3 targetEulerAngles = currentTransform.rotation.eulerAngles;

        targetEulerAngles.z = (targetEulerAngles.z > 180.0f) ? (targetEulerAngles.z - 360.0f) : targetEulerAngles.z;
        targetEulerAngles.z = Mathf.Clamp(targetEulerAngles.z, minRotate, maxRotate);

        currentTransform.rotation = Quaternion.Euler(targetEulerAngles);
    }

    private void LimitSpeed()
    {
        speed = (speed < maxSpeed) ? speed : maxSpeed;
        speed = Mathf.Clamp(speed, minSpeed, maxSpeed);
    }

    private void destroyObjects()
    {
        Destroy(GameObject.Find("Main Camera").GetComponent<AsteroidSpawner>());
        Destroy(GameObject.Find("Flame Left"));
        Destroy(GameObject.Find("Flame Right"));
        Destroy(GameObject.Find("Trail Left"));
        Destroy(GameObject.Find("Trail Right"));
        Destroy(GetComponent<Rigidbody>());
    }

}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Asteroid : MonoBehaviour
{
    public float speed = 6.5f;

    private Rigidbody currentRb;
    private Vector3 screenBounds;
    private Timer scoreData;

    private static int passed = 0;

    void Start()
    {
        scoreData = Camera.main.GetComponent<Timer>();

        currentRb = GetComponent<Rigidbody>();
        currentRb.velocity = new Vector3(0.0f, 0.0f, -speed);
        screenBounds = Camera.main.ScreenToWorldPoint(
            new Vector3(Screen.width, Screen.height, Camera.main.transform.position.z)
            );
    }

    // Update is called once per frame
    void Update()
    {
        if (transform.position.z < screenBounds.z + 15)
        {
            passed++;
            scoreData.currentScore += 5;
            scoreData.passedAsteroids.text = "Passed asteroids: " + passed;
            Destroy(this.gameObject);
        }   
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class Teleport : MonoBehaviour
{
    public GameObject exit;

    private void OnCollisionEnter(Collision myCollision)
    {
        if (myCollision.gameObject.name == "Teleport enter")
        {
            Debug.Log("Entered the teleport");
            this.transform.position = exit.transform.position; 
        }
        if(myCollision.gameObject.name == "New Level")
        {
            SceneManager.LoadScene("Labirint");
        }
    }
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}

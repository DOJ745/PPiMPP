using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TurnLight : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.GetComponent<CapsuleCollider>())
        {
            this.GetComponent<Light>().intensity = 3;
        }
    }

    void OnTriggerStay(Collider other)
    {
        Debug.Log("Someone close! Light On!");
    }
    void OnTriggerExit(Collider other)
    {
        this.GetComponent<Light>().intensity = 0;
    }
}

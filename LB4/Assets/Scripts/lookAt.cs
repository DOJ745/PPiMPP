using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class lookAt : MonoBehaviour
{
    private Renderer renderer;

    public float speed = 1;
    public GameObject followedObj;
    public float closeDistance = 5.0f;
    // Start is called before the first frame update
    void Start()
    {
        renderer = GetComponent<Renderer>();
    }

    // Update is called once per frame
    void Update()
    {
        if(followedObj.transform) {

            Vector3 forward = transform.TransformDirection(Vector3.forward);
            Vector3 toOther = followedObj.transform.position - transform.position;

            if (Vector3.Dot(forward, toOther) < 0)
            {
                print(this.name + " can't find the capsule!");
            }

            Vector3 offset = followedObj.transform.position - transform.position;
            float sqrLen = offset.sqrMagnitude;

            if ( (sqrLen < closeDistance * closeDistance) && (Vector3.Dot(forward, toOther) > 1) )
            {
                print("The capsule is close to " + this.name + ". Chasing him!");
                renderer.material.color = Color.red;
                transform.position = Vector3.MoveTowards(transform.position, followedObj.transform.position, speed * Time.deltaTime);
            }
            else
            {
                renderer.material.color = Color.white;
            }
        }
        
    }
}

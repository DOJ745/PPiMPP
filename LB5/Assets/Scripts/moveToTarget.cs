using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class moveToTarget : MonoBehaviour
{
    // Start is called before the first frame update
    [SerializeField] NavMeshAgent agent;
    public Transform target;
    void Start()
    {
        agent.SetDestination(target.position);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class obstacleMovement : MonoBehaviour
{
    public float speed = .2f;
	public float strength = 9f;

	public bool allowX = true;
	public bool allowY = false;
	public bool allowZ = false;

	private float randomOffset;

	// Use this for initialization
	void Start () {
		randomOffset = Random.Range(0f, 2f);
	}
	
	// Update is called once per frame
	void Update () {
		Vector3 pos = transform.position;

		if(allowX)
		{
			pos.x = Mathf.Sin(Time.time * speed + randomOffset) * strength;
		}

		if(allowY)
		{
			pos.y = Mathf.Sin(Time.time * speed + randomOffset) * strength;
		}
		
		if(allowZ)
		{
			pos.z = Mathf.Sin(Time.time * speed + randomOffset) * strength;
		}
		
		transform.position = pos;
	}
}

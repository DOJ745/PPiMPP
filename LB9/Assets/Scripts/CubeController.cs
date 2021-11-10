using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CubeController : MonoBehaviour
{
    public float maxSpeed = 10f;
    public float jumpForce = 700f;

    bool facingRight = true;
    bool grounded = false;

    public Transform groundCheck;
    public float groundRadius = 0.2f;
    public LayerMask whatIsGround;

    public float move;

    private Rigidbody2D rigidBody;
    // Start is called before the first frame update


    void Start()
    {
        rigidBody = this.GetComponent<Rigidbody2D>();
    }

    // Update is called once per frame
    void Update()
    {
        grounded = Physics2D.OverlapCircle(groundCheck.position, groundRadius, whatIsGround);

        move = Input.GetAxis("Horizontal");


        if (grounded && (Input.GetKeyDown(KeyCode.W) || Input.GetKeyDown(KeyCode.UpArrow)))
        {
            rigidBody.AddForce(new Vector2(0f, jumpForce));
            SoundManagerScript.PlaySound("jump");
        }

        rigidBody.velocity = new Vector2(move * maxSpeed, rigidBody.velocity.y);

        if (move > 0 && !facingRight)
        {
            Flip();
        }
            
        else if (move < 0 && facingRight)
        {
            Flip();
        }
            

        if (Input.GetKey(KeyCode.Escape))
        {
            Application.Quit();
        }

        void Flip()
        {
            SoundManagerScript.PlaySound("walking");
            facingRight = !facingRight;
            Vector3 theScale = transform.localScale;
            theScale.x *= -1;
            transform.localScale = theScale;
        }
    }
}

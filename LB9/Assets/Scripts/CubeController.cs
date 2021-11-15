using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class CubeController : MonoBehaviour
{
    private static int points = 0;

    public float maxSpeed = 10f;
    public float jumpForce = 700f;

    bool facingRight = true;
    bool grounded = false;

    public Transform groundCheck;
    public float groundRadius = 0.2f;
    public LayerMask whatIsGround;

    public float move;
    public Text score;

    public Joystick joystick;

    private Rigidbody2D rigidBody;
    // Start is called before the first frame update
    private void OnCollisionEnter2D(Collision2D coll)
    {
        if (coll.collider == true)
        {
            SoundManagerScript.PlaySound("walking");
        }
        if(coll.collider.gameObject.name == "Coin")
        {
            SoundManagerScript.PlaySound("getCoin");
            Destroy(coll.collider.gameObject);
            points++;
            PlayerScore(points);
        }
    }

    void Start()
    {
        rigidBody = this.GetComponent<Rigidbody2D>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        grounded = Physics2D.OverlapCircle(groundCheck.position, groundRadius, whatIsGround);

        //move = Input.GetAxis("Horizontal");
        move = joystick.Horizontal;


        if (grounded && (Input.GetKeyDown(KeyCode.W) || Input.GetKeyDown(KeyCode.UpArrow) || (joystick.Vertical > 0.35f) ))
        {
            rigidBody.AddForce(new Vector2(0f, jumpForce));
            SoundManagerScript.PlaySound("jump");
        }

        rigidBody.velocity = new Vector2(move * maxSpeed, rigidBody.velocity.y);

        if (move > 0 && !facingRight) { Flip(); }
        else if (move < 0 && facingRight) { Flip(); }
            

        if (points == 5)
        {
            score.text = "You collected all coins!";
            Application.Quit();
        }

        void Flip()
        {
            facingRight = !facingRight;
            Vector3 theScale = transform.localScale;
            theScale.x *= -1;
            transform.localScale = theScale;
        }
    }


    private void PlayerScore(int point)
    {
        Debug.Log("Gained point!");
        score.text = "Score: " + point;
    }
}

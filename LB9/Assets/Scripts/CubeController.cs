using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class CubeController : MonoBehaviour
{
    [SerializeField] private LayerMask platformLayerMask;

    private static int points = 0;

    public float maxSpeed = 10f;
    public float jumpForce = 700f;

    bool facingRight = true;

    public float move;
    public Text score;

    public Joystick joystick;

    private Rigidbody2D rigidBody;

    private void OnCollisionEnter2D(Collision2D coll)
    {
        if (coll.collider == true) {
            Debug.Log(coll.collider);
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
    void Update()
    {
        move = joystick.Horizontal;

        if ( (joystick.Vertical > 0.35f) && IsGrounded() )
        {
            rigidBody.AddForce(new Vector2(0f, jumpForce));
            SoundManagerScript.PlaySound("jump");
        }

        rigidBody.velocity = new Vector2(move * maxSpeed, rigidBody.velocity.y);

        if (move > 0 && !facingRight) { Flip(); }
        else if (move < 0 && facingRight) { Flip(); }
            

        if (points == 5)
        {
            SceneManager.LoadScene("Win");
        }

        void Flip()
        {
            facingRight = !facingRight;
            Vector3 theScale = transform.localScale;
            theScale.x *= -1;
            transform.localScale = theScale;
        }
    }

    private bool IsGrounded()
    {
        Color rayColor;
        float extraHeightTest = 0.1f;

        RaycastHit2D raycastHit = Physics2D.Raycast(
            GetComponent<BoxCollider2D>().bounds.center, 
            Vector2.down, 
            GetComponent<BoxCollider2D>().bounds.extents.y + extraHeightTest,
            platformLayerMask);

        Debug.Log(raycastHit);
        if (raycastHit.collider != null) { rayColor = Color.green; }
        else { rayColor = Color.black; }

        Debug.DrawRay(GetComponent<BoxCollider2D>().bounds.center, 
            Vector2.down * (GetComponent<BoxCollider2D>().bounds.extents.y + extraHeightTest),
            rayColor);

        return raycastHit.collider != null;
    }
    private void PlayerScore(int point)
    {
        Debug.Log("Gained point!");
        score.text = "Score: " + point;
    }
}

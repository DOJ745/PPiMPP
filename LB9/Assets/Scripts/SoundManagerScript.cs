using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SoundManagerScript : MonoBehaviour
{
    public static AudioClip jumpSound, getCoinSound, walkingSound, backgroundMusic;
    static AudioSource AudioSrc;
    // Start is called before the first frame update
    void Start()
    {
        jumpSound = Resources.Load<AudioClip>("jump");
        getCoinSound = Resources.Load<AudioClip>("getCoin");
        walkingSound = Resources.Load<AudioClip>("walking");
        backgroundMusic = Resources.Load<AudioClip>("background");

        AudioSrc = GetComponent<AudioSource>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public static void PlaySound(string clip)
    {
        switch (clip)
        {
            case "jump":
                AudioSrc.PlayOneShot(jumpSound);
                break;

            case "getCoin":
                AudioSrc.PlayOneShot(getCoinSound);
                break;

            case "walking":
                AudioSrc.PlayOneShot(walkingSound);
                break;

            case "background":
                AudioSrc.PlayOneShot(backgroundMusic);
                break;
        }
    }
}

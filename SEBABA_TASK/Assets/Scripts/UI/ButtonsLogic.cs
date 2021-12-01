using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class ButtonsLogic : MonoBehaviour
{
    public Button startButton, highscoresButton, exitButton;

    void Start()
    {
        startButton.onClick.AddListener(startClick);
        highscoresButton.onClick.AddListener(highscoresClick);
        exitButton.onClick.AddListener(exitClick);
    }

    public void startClick()
    {
        Debug.Log("You have clicked the button!");
        SceneManager.LoadScene("Game");
    }

    public void highscoresClick()
    {
        Scores test = new Scores();
        Debug.Log("You have clicked the button!");
        Debug.Log(test.ToString());
    }

    public void exitClick()
    {
        Debug.Log("You have clicked the button!");
    }

}

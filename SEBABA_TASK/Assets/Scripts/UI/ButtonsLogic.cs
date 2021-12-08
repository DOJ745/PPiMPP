using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class ButtonsLogic : MonoBehaviour
{
    public Button startButton, exitButton;

    void Start()
    {
        startButton.onClick.AddListener(startClick);
        exitButton.onClick.AddListener(exitClick);
    }

    public void startClick()
    {
        SceneManager.LoadScene("Game");
    }

    public void exitClick()
    {
        Application.Quit();
    }

}

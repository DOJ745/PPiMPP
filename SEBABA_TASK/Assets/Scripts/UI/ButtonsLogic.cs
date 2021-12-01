using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class ButtonsLogic : MonoBehaviour
{
    public Button startButton, highscoresButton, exitButton;
    private GameDataManager dataManager;

    void Start()
    {
        startButton.onClick.AddListener(startClick);
        highscoresButton.onClick.AddListener(highscoresClick);
        exitButton.onClick.AddListener(exitClick);

        dataManager = GetComponent<GameDataManager>();
        dataManager.writeFile(dataManager.gameData);
    }

    public void startClick()
    {
        Debug.Log("You have clicked the button!");
        SceneManager.LoadScene("Game");
    }

    public void highscoresClick()
    {
        Debug.Log("You have clicked the button!");
        Debug.Log("COLLECTION - " + dataManager.readFile());

        Scores tempData = dataManager.readScores();
        tempData.addScore(500);
        dataManager.writeFile(tempData);

        Debug.Log("ADDED POINTS - " + dataManager.readFile());
    }

    public void exitClick()
    {
        Debug.Log("You have clicked the button!");
    }

}

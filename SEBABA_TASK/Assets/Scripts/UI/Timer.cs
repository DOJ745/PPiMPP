using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class Timer : MonoBehaviour
{
    [Header("Settings")]
    public float timeStart = 0f;

    [Header("Text pannels")]
    public Text textTimer;
    public Text scorePanel;
    public Text highscorePanel;

    [Header("Score data")]
    public GameObject gameDataManagerObj;

    private GameDataManager gameDataManager;
    private int currentScore = 0;
    private int highscore;
    void Start()
    {
        gameDataManager = gameDataManagerObj.GetComponent<GameDataManager>();
        highscore = gameDataManager.readScores().scoreTable.Max();
        highscorePanel.text += " " + highscore.ToString();
        scorePanel.text += " " + currentScore.ToString();
    }

    void Update()
    {
        timeStart += Time.deltaTime;

        int seconds = (int)(timeStart % 60);
        int minutes = (int)(timeStart / 60) % 60;

        currentScore = seconds;

        textTimer.text = string.Format("{0:00}:{1:00}", minutes, seconds);
        scorePanel.text = "Score: " + string.Format("{0}", currentScore);
    }
}

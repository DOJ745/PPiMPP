using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class Timer : MonoBehaviour
{
    [Header("Settings")]
    public float timeStart = 0f;
    public int currentScore = 0;

    [Header("Text pannels")]
    public Text textTimer;
    public Text scorePanel;
    public Text highscorePanel;
    public Text passedAsteroids;

    [Header("Score data")]
    public GameObject gameDataManagerObj;

    private GameDataManager gameDataManager;

    private int highscore;
    void Start()
    {
        gameDataManager = gameDataManagerObj.GetComponent<GameDataManager>();

        highscore = gameDataManager.readScores().scoreTable.Max();
        highscorePanel.text += " " + highscore.ToString();
        scorePanel.text += " " + currentScore.ToString();

        InvokeRepeating("AddScore", 0.0f, 1.0f);
    }

    void Update()
    {
        timeStart += Time.deltaTime;

        int seconds = (int)(timeStart % 60);
        int minutes = (int)(timeStart / 60) % 60;

        textTimer.text = string.Format("{0:00}:{1:00}", minutes, seconds);
        scorePanel.text = "Score: " + currentScore;

        if (currentScore > highscore)
        {
            highscore = currentScore;
            highscorePanel.text = "! NEW Highscore ! : " + highscore;
        }
            
    }

    private void AddScore()
    {
        if (Input.GetKey(KeyCode.Space)) { currentScore += 2; }
        else { currentScore += 1; }
    }
}

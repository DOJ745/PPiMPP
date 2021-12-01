using UnityEngine;
using System.IO;

public class GameDataManager : MonoBehaviour
{
    public string saveFile;

    public Scores gameData = new Scores();
    void Awake()
    {
        // Update the path once the persistent path exists.
        saveFile = Application.persistentDataPath + "/gamedata.json";
    }

    public Scores readScores()
    {
        if (File.Exists(saveFile))
        {
            string fileContents = File.ReadAllText(saveFile);

            gameData = JsonUtility.FromJson<Scores>(fileContents);
        }
        return gameData;
    }

    public string readFile()
    {
        if (File.Exists(saveFile))
        {
            string fileContents = File.ReadAllText(saveFile);

            gameData = JsonUtility.FromJson<Scores>(fileContents);
        }

        return gameData.ToString();
    }

    public void writeFile(Scores _gameData)
    {
        string jsonString = JsonUtility.ToJson(_gameData);

        File.WriteAllText(saveFile, jsonString);
    }
}

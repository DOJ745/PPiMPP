using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

[Serializable]
public class Scores
{
    public List<int> scoreTable;

    public Scores() 
    { 
        scoreTable = new List<int>();
        fillScores(10);
    }

    public void fillScores(int tableSize)
    {
        for (int i = 0; i < tableSize; i++)
        {
            scoreTable.Add(100 + i);
        }

        scoreTable.Sort((x, y) => Comparer.Default.Compare(y, x));
    }

    public void addScore(int _score)
    {
        int minElem = scoreTable.Min();

        if (scoreTable.Remove(minElem))
        {
            scoreTable.Add(_score);
            scoreTable.Sort((x, y) => Comparer.Default.Compare(y, x));
        }
     
    }

    public override string ToString()
    {
        string temp = "Scores collection:\n";

        for (int i = 0; i < scoreTable.Count; i++)
        {
            temp += scoreTable[i] + "\n";
        }

        return temp;
    }
}

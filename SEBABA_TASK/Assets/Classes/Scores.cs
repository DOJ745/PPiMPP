using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class Scores
{
    public ArrayList scores { get; set; }

    public Scores() 
    { 
        scores = new ArrayList();
        fillScores(10);
    }

    public void saveScores()
    {
        string result = JsonUtility.ToJson(scores);
    }

    public ArrayList readScores(string collecttion)
    {
        Scores result = JsonUtility.FromJson<Scores>(collecttion);
        return result.scores;
    }

    public void fillScores(int tableSize)
    {
        for (int i = 0; i < tableSize; i++)
        {
            scores.Add(100 + i);
        }

        scores.Sort(new ReverseSort());
    }

    public override string ToString()
    {
        string temp = "";

        for (int i = 0; i < scores.Count; i++)
        {
            temp += scores[i] + "\n";
        }

        return temp;
    }

    private class ReverseSort : IComparer
    {
        public int Compare(object x, object y)
        {
            return Comparer.Default.Compare(y, x);
        }
    }
}

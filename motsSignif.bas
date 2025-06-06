Attribute VB_Name = "Module1"
Sub ExtraireMotsSignificatifs_Dynamique()
    Dim ws As Worksheet
    Dim headerRow As Long
    Dim lastCol As Long, lastRow As Long
    Dim hopCol As Long, motsCol As Long
    Dim i As Long, c As Long
    Dim stopwords As Variant
    Dim nomHopital As String, tempStr As String
    Dim mots As Variant, mot As Variant
    Dim resultStr As String
    
    ' Référencer la feuille active (ou remplacer par Worksheets("NomFeuille"))
    Set ws = ActiveSheet
    headerRow = 1
    
    ' 1) Déterminer la dernière colonne remplie (en-têtes)
    lastCol = ws.Cells(headerRow, ws.Columns.Count).End(xlToLeft).Column
    
    ' 2) Chercher "Nom hopital" et "Mots significatif" dans la ligne d'en-tête
    hopCol = 0: motsCol = 0
    For c = 1 To lastCol
        Select Case Trim(ws.Cells(headerRow, c).Value)
            Case "Nom hopital"
                hopCol = c
            Case "Mots significatif"
                motsCol = c
        End Select
    Next c
    
    ' 3) Si "Nom hopital" non trouvé, arrêter
    If hopCol = 0 Then
        MsgBox "Colonne ""Nom hopital"" introuvable.", vbCritical
        Exit Sub
    End If
    
    ' 4) Si "Mots significatif" n'existe pas, insérer colonne à droite de "Nom hopital"
    If motsCol = 0 Then
        ws.Columns(hopCol + 1).Insert Shift:=xlToRight
        ws.Cells(headerRow, hopCol + 1).Value = "Mots significatif"
        motsCol = hopCol + 1
    End If
    
    ' 5) Définir les mots génériques à retirer (en MAJUSCULE pour comparaison)
    stopwords = Array("GRAND", "CHU", "CHI", "CH", "GBU", "HÔPITAL", "HOPITAL", "CLINIQUE", _
                      "MATERNITÉ", "MATERNITE", "HCL", "GH")
    
    ' 6) Déterminer la dernière ligne de données de la colonne "Nom hopital"
    lastRow = ws.Cells(ws.Rows.Count, hopCol).End(xlUp).Row
    
    ' 7) Boucler sur chaque ligne pour extraire les mots significatifs
    For i = headerRow + 1 To lastRow
        nomHopital = CStr(ws.Cells(i, hopCol).Value)
        If Len(Trim(nomHopital)) = 0 Then
            ws.Cells(i, motsCol).Value = ""
        Else
            ' a) Remplacer tirets, virgules, parenthèses, slashs… par un espace
            tempStr = nomHopital
            tempStr = Replace(tempStr, "-", " ")
            tempStr = Replace(tempStr, "/", " ")
            tempStr = Replace(tempStr, ",", " ")
            tempStr = Replace(tempStr, "(", " ")
            tempStr = Replace(tempStr, ")", " ")
            tempStr = Replace(tempStr, "_", " ")
            ' (ajoutez d'autres remplacements si besoin)
            
            ' b) Split en mots
            mots = Split(Application.WorksheetFunction.Trim(tempStr), " ")
            
            ' c) Parcourir chaque mot : si ce n'est PAS un stopword, on le conserve
            resultStr = ""
            For Each mot In mots
                If Len(Trim(mot)) > 0 Then
                    If IsError(Application.Match(UCase(mot), stopwords, 0)) Then
                        ' Le mot n'est pas dans la liste des stopwords ? on le garde
                        resultStr = resultStr & mot & " "
                    End If
                End If
            Next mot
            
            ' d) Supprimer le dernier espace superflu
            resultStr = Trim(resultStr)
            
            ' e) Écrire le résultat dans la colonne "Mots significatif"
            ws.Cells(i, motsCol).Value = resultStr
        End If
    Next i
    
    MsgBox "Extraction des mots significatifs terminée pour " & (lastRow - headerRow) & " lignes.", vbInformation
End Sub



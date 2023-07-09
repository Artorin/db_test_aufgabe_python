Python 3.10, funktioniert auch auf 3.9 

In der Methode compute_shortest_paths() werden 4 methoden aufgerufen:
	change_struct_graph()
	dijkstra()
	find_another()
	find_in_dijkstra()

	change_struct_graph() - wir konzentrieren uns auf Knoten, ändern die Darstellung des Graph 
	
	dijkstra() - berechnet ein kurzester Pfad in der Graph - uneffiktiv weil Algorithm ist greedy, aber leicht zu implementieren (besser A* aber braucht man Heuristic ausrechnen)
	
	find_another() - hier werden jegliche Wiederholungen berechnet, die aber immer zu einem Knoten führen 
			
			es hat nicht die Zeit gereicht um die Wiederholungen zu finden, die drinnen n Knoten haben 
			
			auch es kann noch nicht alle ausgehende Pfade aus dem nächten Knoten zu gehen - muss man noch weiter debuggen  
			
			noch werden die Schleifen nicht gefunden, die aus den und in die Endknoten führen 

	find_in_dijkstra() - ist eine vereinfachste Variante vom Yen's Algorithm (works on assumption that another shortest paths are near from the shortest path), löscht immer ein Node in dem kürzesten Pfad und sucht nach dem neuen Pfad. 
				
			allerdings Algorithm hier ist uneffektiv, weil falls tolerance n > 1 der schmeisst die Pfade weg, die grösser als len_kurzeste*n

	Habe versucht auch Yen's Algorithm zu nutzen, aber der findet nur n Pfaden, die an tolerance wert nicht passen


An sich beste Lösung wäre A* algorithm zu nutzen um den ersten kürzesten Pfad zu finden, aber müsste man überlegen wie eine gute Heuristic zu bekommen. Auch find_another() method müsste man so erweitern, dass es im Prinzip wie ein Yen's Algorithm verhält. 
und find_in_dijkstra() dann wird nicht mehr gebraucht. 

Beste Grüße 
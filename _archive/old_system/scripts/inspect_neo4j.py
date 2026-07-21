from __future__ import annotations
from src.config import get_config
from src.neo4j_adapter import Neo4jMemoryAdapter


def main() -> None:
    config = get_config()
    adapter = Neo4jMemoryAdapter(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
    try:
        print('Ping:', adapter.ping())
        print('Counts:', adapter.smoke_counts())
        print('\nSample episodic memories:')
        for r in adapter.search_episodic('user01', 'project', limit=5):
            print(r)
        print('\nSample semantic facts:')
        for r in adapter.search_semantic('user01', 'project', limit=5):
            print(r)
    finally:
        adapter.close()

if __name__ == '__main__':
    main()

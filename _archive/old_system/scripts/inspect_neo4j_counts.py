from src.config import get_config
from src.neo4j_adapter import Neo4jMemoryAdapter


def main():
    config = get_config()

    adapter = Neo4jMemoryAdapter(
        config.neo4j_uri,
        config.neo4j_user,
        config.neo4j_password,
    )

    try:
        print("Ping:", adapter.ping())
        print("Counts:", adapter.smoke_counts())
    finally:
        adapter.close()


if __name__ == "__main__":
    main()